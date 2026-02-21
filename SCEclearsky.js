/**** set your roi ****/
var roi = ee.FeatureCollection("projects/ee-yjiawei72/assets/watershed_gongga");

Map.centerObject(roi, 9);
var ndsiThresh = 0.4;
var CLEAR_THRESHOLD = 0.55;
// ================= Utility functions =================
function applyScaleFactorsSen(image) {
  var opticalBands = image.select(['B2','B3','B4','B8','B11','B12']).divide(10000);
  return image.addBands(opticalBands, null, true);
}

function addNDSI(image) {
  var ndsi = image.normalizedDifference(['B3','B11']).rename('NDSI');   //NDSI  B3,B11
  return image.addBands(ndsi);
}

function addNDSI(image) {
  var ndsi = image.normalizedDifference(['B3','B11']).rename('NDSI');   //NDSI  B3,B11
  return image.addBands(ndsi);
}

var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var QA_BAND = 'cs_cdf';

var currentCollection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2019-01-01','2025-07-18') 
  .linkCollection(csPlus,[QA_BAND])
  .map(function(img){
    return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
  })
  .map(applyScaleFactorsSen)
  .map(addNDSI);
print(currentCollection,'test')
var current = currentCollection.median().clip(roi);

var ndsiClear = current.select('NDSI').gte(ndsiThresh).clip(roi);

Map.addLayer(ndsiClear, {min: 0,max: 1,palette: ['#D9D9D9', '#4A90E2'] }, 'Snow Final (mask)');

Export.image.toDrive({
   image: ndsiClear,
   description: "snow",  
   folder: 'snow1',      
   region:roi,
   scale:10,
   maxPixels: 1e13,
});
