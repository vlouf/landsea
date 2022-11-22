#include <ctime>
#include <iostream>
#include <filesystem>

#include <bom/io/configuration.h>
#include <bom/io/cf.h>
#include <bom/io/nc.h>
#include <bom/io/odim.h>
#include <bom/radar/beam_propagation.h>
#include <bom/array2.h>
#include <bom/array1.h>
#include <bom/ellipsoid.h>
#include <bom/grid_coordinates.h>
#include <bom/grid_transform.h>
#include <bom/map_projection.h>
#include <bom/trace.h>

using namespace bom;

constexpr float nodata = std::numeric_limits<float>::quiet_NaN();
constexpr float undetect = -32.0f;

struct bin_info
{
  float slant_range;
  float ground_range;
  float altitude;
};

struct sweep
{
  radar::beam_propagation beam;
  array1<bin_info>        bins; // @ bin centers
  array1<angle>           rays; // @ ray centers
  array2f                 data;
};

struct volume
{
  latlonalt     location;
  vector<sweep> sweeps;
};

struct radarset
{
  volume vradh;
  volume dbzh;
  array1f nyquist;
  array1f elevation;
  string source;
  string date;
  string time;
};

struct seamask{
    array1f lat;
    array1f lon;
    array2i mask;
    seamask(vec2z shape) : lat{shape.y}, lon{shape.x}, mask{shape} { }
};

auto argmin(array1f const& x) -> int{
  // Determines the location of the element in the array with the minimum value
  int pos = std::distance(x.begin(), std::min_element(x.begin(), x.end()));
  return pos;
}

auto read_bins(io::odim::dataset const& scan_odim, radar::beam_propagation const& beam) -> array1<bin_info>
{
  auto bins = array1<bin_info>(scan_odim.attributes()["nbins"].get_integer());
  auto range_scale = scan_odim.attributes()["rscale"].get_real();
  auto range_start = scan_odim.attributes()["rstart"].get_real() * 1000.0 + range_scale * 0.5;
  for (size_t i = 0; i < bins.size(); ++i)
  {
    bins[i].slant_range = range_start + i * range_scale;
    std::tie(bins[i].ground_range, bins[i].altitude) = beam.ground_range_altitude(bins[i].slant_range);
  }
  return bins;
}

auto read_rays(io::odim::dataset const& scan_odim) -> array1<angle>
{
  auto rays = array1<angle>(scan_odim.attributes()["nrays"].get_integer());
  auto istartaz = scan_odim.attributes().find("startazA");
  auto istopaz = scan_odim.attributes().find("stopazA");
  if (istartaz != scan_odim.attributes().end() && istopaz != scan_odim.attributes().end())
  {
    auto starts = istartaz->get_real_array();
    auto stops = istopaz->get_real_array();
    for (size_t i = 0; i < rays.size(); ++i)
      rays[i] = lerp(starts[i], stops[i], 0.5) * 1_deg;
  }
  else if (scan_odim.attributes()["product"].get_string() == "SCAN")
  {
    auto ray_scale = 360_deg / rays.size();
    auto ray_start = scan_odim.attributes().find("astart") != scan_odim.attributes().end()
      ? scan_odim.attributes()["astart"].get_real() * 1_deg
      : 0.0_deg;
    ray_start += ray_scale * 0.5;
    for (size_t i = 0; i < rays.size(); ++i)
      rays[i] = ray_start + i * ray_scale;
  }
  else
    throw std::runtime_error{"Unable to determine scan azimuth angles"};
  return rays;
}

auto read_moment(io::odim::file const vol_odim, string moment) -> volume{
  auto vol = volume{};

  vol.location.lat = vol_odim.attributes()["lat"].get_real() * 1_deg;
  vol.location.lon = vol_odim.attributes()["lon"].get_real() * 1_deg;
  vol.location.alt = vol_odim.attributes()["height"].get_real();

  for (size_t iscan = 0; iscan < vol_odim.dataset_count(); ++iscan)
  {
    auto scan_odim = vol_odim.dataset_open(iscan);
    auto scan = sweep{};

    auto elangle = scan_odim.attributes()["elangle"].get_real();

    if(std::fabs(elangle - 90) < 0.1f)
    {
      // std::cout << "Skipping 90 deg scan" << std::endl;
      continue;
    }

    scan.beam = radar::beam_propagation{vol.location.alt, elangle * 1_deg};
    scan.bins = read_bins(scan_odim, scan.beam);
    scan.rays = read_rays(scan_odim);

    for (size_t idata = 0; idata < scan_odim.data_count(); ++idata)
    {
      auto data_odim = scan_odim.data_open(idata);
      if (data_odim.quantity() != moment)
        continue;

      scan.data.resize(vec2z{scan.bins.size(), scan.rays.size()});
      if((moment.compare("VRADH") != 0) || moment.compare("VRADDH") != 0)
      {
        data_odim.read_unpack(scan.data.data(), nodata, nodata);
      }
      else
      {
        data_odim.read_unpack(scan.data.data(), undetect, nodata);
      }

      vol.sweeps.push_back(std::move(scan));
      break;
    }
  }

  return vol;
}

auto read_global_seamask(string const filename) -> seamask{

  auto dset = io::nc::file{filename, io_mode::read_only};
  size_t nx = dset.lookup_dimension("longitude").size();
  size_t ny = dset.lookup_dimension("latitude").size();

  auto landsea = seamask{vec2z{nx, ny}};
  dset.lookup_variable("latitude").read(landsea.lat);
  dset.lookup_variable("longitude").read(landsea.lon);
  dset.lookup_variable("elevation").read(landsea.mask);

  return landsea;
}

auto check_is_ocean(seamask const& landsea, latlon loc) -> bool{
  auto plat = array1f{landsea.lat.size()};
  auto plon = array1f{landsea.lon.size()};
  for(size_t i=0; i<landsea.lat.size(); i++){
    plat[i] = std::fabs(landsea.lat[i] - loc.lat.degrees());
  }
  for(size_t i=0; i<landsea.lon.size(); i++){
    plon[i] = std::fabs(landsea.lon[i] - loc.lon.degrees());
  }

  int ilon = argmin(plon);
  int ilat = argmin(plat);

  return landsea.mask[ilat][ilon] < 0;
}

auto read_refl_corrected(std::filesystem::path const& path, io::configuration const& config) -> volume{
  // Read radar file
  auto vol_odim = io::odim::file{path, io_mode::read_only};
  auto dbzh = read_moment(vol_odim, "DBZH");
  auto dbzh_clean = read_moment(vol_odim, "DBZH_CLEAN");

  // Read seamask.
  auto landsea = read_global_seamask(config["topography"]);
  int count=0;
  auto radarloc = latlon{dbzh.location.lon, dbzh.location.lat};

  for(size_t i=0; i<dbzh.sweeps.size(); i++){
    for(size_t j=0; j<dbzh.sweeps[i].bins.size(); j++){
      for(size_t k=0; k<dbzh.sweeps[k].bins.size(); k++){
        auto r0 = dbzh.sweeps[i].data[j][k];
        auto r1 = dbzh_clean.sweeps[i].data[j][k];
        if(!(std::isnan(r0) | (r0 == undetect))){         
          if(!(std::isnan(r1) | (r1 == undetect)))
            continue;

          auto gate_latlon = wgs84.bearing_range_to_latlon(
            radarloc,
            dbzh.sweeps[i].rays[k],
            dbzh.sweeps[i].bins[j].ground_range
          );
          if(check_is_ocean(landsea, gate_latlon)){
            dbzh.sweeps[i].data[j][k] = nodata;            
          }
        }
      }
    }
  }
  std::cout << "Corrected " << count << " points over sea." << std::endl;

  return dbzh;
}

int main(){
    puts("Hello");
    string filename = "/srv/data/swirl/vols/2/2_20221121_003500.pvol.h5";
    string configfile = "/srv/data/swirl/config/flow.2.conf";
    auto config = io::configuration{std::ifstream{configfile}};
    auto vol = read_refl_corrected(filename, config);
    puts("odim radar file read");
    return 0;
}
