mapboxgl.accessToken = 'pk.eyJ1IjoiaGV0aW5naGVsZW4iLCJhIjoiY2w2b2hmZ2R4MGFyczNiczNlaWd3cXAyMyJ9.gCRtPIhJMjSNlW67i-iozA'; // default public token
var map = new mapboxgl.Map({
  container: 'map',
  //style: 'mapbox://styles/hetinghelen/cl6ogwalx000214qq3wgej6po',
  style: 'mapbox://styles/mapbox/light-v10',
  zoom: 6.5,
  maxZoom: 9,
  minZoom: 3,
  center: [-85.5, 37.7],
});

map.on("load", function () {
  map.addLayer(
    {
      id: "us_states_elections_outline",
      type: "line",
      source: {
        type: "geojson",
        data: "data/statesElections.geojson",
      },
      paint: {
        "line-color": "#ffffff",
        "line-width": 0.7,
      },
    },
    "waterway-label" // Here's where we tell Mapbox where to slot this new layer
  ); 
  map.addLayer(
    {
      id: "us_states_elections",
      type: "fill",
      source: {
        type: "geojson",
        data: "data/statesElections.geojson",
      },
      paint: {
        "fill-color": [
          "match",
          ["get", "Winner"],
          "Donald J Trump", "#cf635d",
          "Joseph R Biden Jr", "#6193c7",
          "Other", "#91b66e",
          "#ffffff",
        ],
        "fill-outline-color": "#ffffff",
        "fill-opacity": [
            "step",
            ["get", "WnrPerc"],
            0.3,
            0.4,
            0.5,
            0.5,
            0.7,
            0.6,
            0.9,
          ],
      },
    },
    "us_states_elections_outline"
  );
});