import ee
import datetime
import time
import traceback

# Authenticate and initialize Earth Engine
ee.Initialize()

def wood_stock_analysis():
    # Define Area of Interest (AOI)
    aoi = ee.FeatureCollection("projects/ee-turkimdhia/assets/cogra_sites")

    # Define dynamic date range
    end_date = ee.Date(datetime.datetime.now().strftime('%Y-%m-%d'))
    start_date = end_date.advance(-12, 'day')

    # Updated Sentinel-2 Dataset
    sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterDate(start_date, end_date) \
        .filterBounds(aoi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    # Check if there are any images in the collection
    collection_size = sentinel2.size().getInfo()
    if collection_size == 0:
        print("No suitable Sentinel-2 images found in the specified date range.")
        return None

    # Preprocessing function
    def preprocess(image):
        # Use image metadata to set timestamp if available
        return image.clip(aoi) \
            .select(['B2', 'B3', 'B4', 'B8']) \
            .divide(10000) \
            .set('system:time_start', image.get('system:time_start'))

    # Process Sentinel-2 collection
    processed = sentinel2.map(preprocess)

    # Define geometries for training data
    wood_craponne = ee.Geometry.MultiPolygon([
        [[[3.833539188197541, 45.318438111152155],
        [3.833388984492707, 45.31824196286186],
        [3.833796680262971, 45.31807599070126],
        [3.8340756300005197, 45.31830231625427]]],
        [[[3.8336406941989054, 45.31742342310926],
        [3.833420753059684, 45.317412106677565],
        [3.8334153886416544, 45.317272537167376],
        [3.8336192365267863, 45.317268765013665],
        [3.833651423034965, 45.31732157514255]]],
        [[[3.833222269592582, 45.31815521422775],
        [3.8329325910189738, 45.31799301391878],
        [3.8332598205187907, 45.317778003491235],
        [3.833501219330131, 45.3180307349623]]],
        [[[3.832852124748527, 45.31778177561102],
        [3.832573175010978, 45.31778177561102],
        [3.832696556625663, 45.31758562504868],
        [3.832954048691093, 45.31760448570919]]],
        [[[3.8326917465438104, 45.31849939952606],
        [3.832530814002917, 45.31830325144792],
        [3.8326702888716913, 45.318159912038354],
        [3.8328848655928827, 45.318159912038354],
        [3.8328634079207635, 45.31837114893645]]]
    ])

    wood_severac = ee.Geometry.MultiPolygon([
        [[[3.060462827908621, 44.33238573400508],
        [3.0602804376956083, 44.332262944862904],
        [3.060387726056204, 44.33210945807362],
        [3.06074177764617, 44.33212480677064],
        [3.0607310488101103, 44.33230131649746]]],
        [[[3.060763235318289, 44.33185620399314],
        [3.0605808451052763, 44.331777541518306],
        [3.0605969383593656, 44.3316374836799],
        [3.0612057998057463, 44.33164323948808],
        [3.0612031175967314, 44.33184852961]]],
        [[[3.061288948285208, 44.33217469000814],
        [3.060913439023123, 44.33218620151848],
        [3.0609241678591825, 44.33196364525177],
        [3.061315770375357, 44.331944459327154],
        [3.061267490663089, 44.33211905091091]]],
        [[[3.060847363773771, 44.33227108650925],
        [3.0611906865276772, 44.332282598000646],
        [3.061192388760672, 44.332431779867136],
        [3.0607525064822294, 44.33237038537636]]]
    ])

    notWood_craponne = ee.Geometry.MultiPolygon([
        [[[3.8346915015573124, 45.3183711682744],
        [3.8342945346231083, 45.31809957783233],
        [3.8346378573770146, 45.317873251469535],
        [3.8350670108193974, 45.31816747556489]]],
        [[[3.833725906311951, 45.3187483750622],
        [3.835152841507874, 45.318537139570395],
        [3.835345960556946, 45.31898978608855],
        [3.833908296524964, 45.31914066745769]]],
        [[[3.8338117370004277, 45.31699057001686],
        [3.8350133666390995, 45.31674915047283],
        [3.83524940103241, 45.31719426695712],
        [3.8340585002297978, 45.317367785995316]]]
    ])

    notWood_severac = ee.Geometry.MultiPolygon([
        [[[3.062764163243399, 44.331894576283126],
        [3.0614767029162504, 44.33187539033591],
        [3.0614981605883695, 44.33143794903686],
        [3.0627748920794584, 44.33145713512717]]],
        [[[3.0626515104647734, 44.33226294525218],
        [3.061712737309561, 44.33225910808735],
        [3.0616805508013822, 44.33197899437623],
        [3.062662239300833, 44.33196364564104]]],
        [[[3.061291630494223, 44.33247974459935],
        [3.0613184525843717, 44.33232625837748],
        [3.061763699280844, 44.332291723922204],
        [3.0617690636988737, 44.33240300153863]]]
    ])

    # Create FeatureCollections
    wood_stock = ee.FeatureCollection([
        ee.Feature(wood_craponne, {'class': 1}),
        ee.Feature(wood_severac, {'class': 1})
    ])

    not_wood = ee.FeatureCollection([
        ee.Feature(notWood_craponne, {'class': 0}),
        ee.Feature(notWood_severac, {'class': 0})
    ])

    training_data = wood_stock.merge(not_wood)

    # Median composite for training
    training_image = processed.median()

    # Sampling the training image
    training_samples = training_image.sampleRegions(
        collection=training_data,
        properties=['class'],
        scale=10
    )

    # Train the classifier
    classifier = ee.Classifier.smileRandomForest(10).train(
        features=training_samples,
        classProperty='class',
        inputProperties=['B2', 'B3', 'B4', 'B8']
    )

    # Classify and process results for each image in the collection
    def classify_and_calculate(image):
        classified_image = image.classify(classifier)
        wood_stock_per_site = classified_image.eq(1).multiply(ee.Image.pixelArea()).reduceRegions(
            collection=aoi,
            reducer=ee.Reducer.sum(),
            scale=10
        )
        return wood_stock_per_site.map(lambda feature: feature.set({
            'timestamp': image.date().format('YYYY-MM-dd'),
            'woodStockArea': ee.Number(feature.get('sum')),
            'siteName': feature.get('name')  # get name from aoi features
        }))

    # Map the function over the processed collection and flatten results
    results = processed.map(classify_and_calculate).flatten()

    # Export results to Google Drive
    task = ee.batch.Export.table.toDrive(
        collection=results,
        description=f'AlphaEO/WoodStockAnalysis_auto_{datetime.datetime.now().strftime("%Y%m%d")}',
        fileFormat='CSV'
    )
    task.start()
    print("Export task started:", task.status())

    # Monitor the task
    while task.active():
        print("Task status:", task.status())
        time.sleep(10)

    print("Task finished with status:", task.status())
    return task

# Main execution with error handling
try:
    # Run the wood stock analysis
    task_result = wood_stock_analysis()
    
    if task_result is None:
        print("No task was started due to no available images.")
except Exception as e:
    print("An error occurred:")
    print(traceback.format_exc())