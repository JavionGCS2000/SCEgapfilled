/**** set your ROI ****/
var roi = ee.FeatureCollection("projects/ee-yjiawei72/assets/watershed_gongga");

// ======================= parameters =======================
var ndsiThresh = 0.4;
var trainSampleCount = 5000;
var rfNumTrees = 20; 
var CLEAR_THRESHOLD = 0.55;
// ================================================================

Map.centerObject(roi, 9);

// ================= Utility functions =================
function applyScaleFactorsSen(image) {
  var opticalBands = image.select(['B2','B3','B4','B8','B11','B12']).divide(10000);
  return image.addBands(opticalBands, null, true);
}


function addNDSI(image) {
  var ndsi = image.normalizedDifference(['B3','B11']).rename('NDSI');   //NDSI  B3,B11
  return image.addBands(ndsi);
}

// ================= Compute snow frequency (IF) =================
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var QA_BAND = 'cs_cdf';

// Build a clear S2 collection for historical computing
var s2_clear = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2019-01-01', '2025-07-18')
  .linkCollection(csPlus, [QA_BAND])
  .map(function(img){
    return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
  })
  .map(applyScaleFactorsSen)
  .map(addNDSI);
print(s2_clear,'s2_clear')
// Compute frequency of NDSI >= ndsiThresh
function computeSnowFreq(collection, ndsiThresh){
  var snowMaskCol = collection.map(function(img){
    return img.select('NDSI').gte(ndsiThresh)
              .updateMask(img.select('NDSI').mask())
              .rename('snow'); 
  });
  // mean of binary snow masks -> frequency [0,1]
  return snowMaskCol.mean().rename('snow_freq').clip(roi);
}

var snow_freq = computeSnowFreq(s2_clear, ndsiThresh);
Map.addLayer(snow_freq, {min:0, max:1, palette:['red','orange','blue']}, 'Snow Frequency (IF)',false);

// ================= Define permanent/fluctuation zones =================
var permSnow = snow_freq.gte(0.95).rename('permSnow');   
var permNoSnow = snow_freq.eq(0).rename('permNoSnow'); 
var fluctZone = snow_freq.gt(0).and(snow_freq.lt(0.95)).rename('fluct'); 

Map.addLayer(permSnow.updateMask(permSnow), {}, 'Permanent Snow (mask)',false);
Map.addLayer(permNoSnow.updateMask(permNoSnow), {}, 'Permanent NoSnow (mask)',false);
Map.addLayer(fluctZone.updateMask(fluctZone), {palette:['yellow']}, 'Fluctuation Zone',false);

// ================= Prepare current (target) image and cloud mask =================
var currentCollection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2022-04-09','2022-04-10') 
  .linkCollection(csPlus,[QA_BAND])
  .map(applyScaleFactorsSen)
  .map(addNDSI);
//print(currentCollection,'img')

var current = currentCollection.median().clip(roi);
var cloudMask = current.select(QA_BAND).lt(CLEAR_THRESHOLD).rename('cloud'); 
//Map.addLayer(cloudMask, {min:0, max:1, palette:['white','gray']}, 'Cloud Mask');

// ================= Elevation: resample/reproject to match target =================
var srtm = ee.Image('NASA/NASADEM_HGT/001');
var ele = ee.Terrain.products(srtm).select('elevation').rename('elevation');

var targetProj = current.select('B4').projection(); // use band B4's projection (10 m)
ele = ele.reproject({crs: targetProj.crs(), scale: 10});

// ================= Build feature image (snow_freq + elevation) =================
var featureImage = snow_freq.addBands(ele).clip(roi);

// Masked zones for clear vs cloudy within fluctuation zone
var clearFluct = fluctZone.updateMask(cloudMask.not()); 
var cloudFluct = fluctZone.updateMask(cloudMask);     

// Observed snow now for clear fluctuation pixels (binary)
var ndsiClearFluct = current.select('NDSI').updateMask(clearFluct);
var snowClear = ndsiClearFluct.gte(ndsiThresh).rename('snow_now').unmask(0).clip(roi); 
//Map.addLayer(snowClear.updateMask(snowClear), {palette:['cyan']}, 'Observed Snow in Clear Fluctuation');

// ================= TRAINING and GAP-FILLING =================
var trainStack = featureImage.addBands(snowClear).updateMask(clearFluct);

// Use stratifiedSample to get balanced samples from classes present
var training = trainStack.select(['snow_freq','elevation','snow_now'])
  .stratifiedSample({
    region: roi,
    classBand: 'snow_now',
    scale: 10,
    numPoints: trainSampleCount,
    seed: 42,
    geometries: false
  }).filter(ee.Filter.notNull(['snow_now']));

// Check if there are both classes in training (defensive)
var classCounts = training.aggregate_histogram('snow_now');
//print('Training class counts', classCounts);

// Train Random Forest (classification)
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: rfNumTrees,
  seed: 1
}).train({
  features: training,
  classProperty: 'snow_now',
  inputProperties: ['snow_freq', 'elevation']
});

// Predict only on fluctuation zone (boundary condition: only predict where snow_freq > 0)
var featureStackForPred = featureImage.updateMask(fluctZone).updateMask(snow_freq.gt(0));
var rfClassified = featureStackForPred.classify(rf).select('classification').rename('rf_raw');

var rfCleaned = rfClassified.rename('rf_gapfilled');

// ================= Combine outputs: permSnow / permNoSnow / observed / gapfilled =================
// Start from an empty image
var snow_final = ee.Image(0).rename('snow_final');

// 1) permanent snow -> 1
snow_final = snow_final.where(permSnow.eq(1), 1);

// 2) permanent no snow -> 0 (already 0 by default, but explicit)
snow_final = snow_final.where(permNoSnow.eq(1), 0);

// 3) in fluctuation: if clear -> take observed snowClear
snow_final = snow_final.where(clearFluct.eq(1), snowClear);

// 4) in fluctuation & cloudy -> take gapfilled prediction
snow_final = snow_final.where(cloudFluct.eq(1), rfCleaned);

// Clip and set properties
snow_final = snow_final.clip(roi).set({'description': 'snow_final_combined'});

// ================= Visualization =================
Map.addLayer(featureStackForPred.select('snow_freq'), {min:0, max:1, palette:['white','blue']}, 'Pred mask snow_freq (fluct zone)',false);
Map.addLayer(rfClassified.updateMask(fluctZone), {min:0, max:1, palette:['red','blue']}, 'RF raw (fluct zone)',false);
Map.addLayer(rfCleaned.updateMask(fluctZone), {min:0, max:1, palette:['yellow','green']}, 'RF cleaned (fluct zone)',false);
Map.addLayer(snow_final, {min: 0,max: 1,palette: ['#D9D9D9', '#4A90E2'] }, 'Snow Final (mask)');

// Optional: compare original clear observation vs final
Map.addLayer(snowClear.updateMask(snowClear), {palette:['cyan']}, 'Observed Snow (clear only)',false);

Export.image.toDrive({
   image: snow_final,
   description: "snow",  
   region:roi,
   scale:10,
   maxPixels: 1e13,
});

