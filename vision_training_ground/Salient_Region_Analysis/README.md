# Sareana: Salient Region Analysis
The intent of this analysis is to prioritize salient regions of the world for use with visual satellite navigation. 

The process is as follows:
1. ```make_grid_ims.py``` to section the Blue Marble world images in the bm1k folder into 1 image for each Military Grid Reference System (MGRS) region. 

2. ```get_maps.py``` to first create saliency maps for each region image using ```cv2.StaticSaliencyFineGrained()```, and then to merge all of the saliency maps back together into an image of the full world:
![merged saliency map of the world](world_saliency.jpg)

3. ```get_maps.py``` to consolidate world cloud maps in cloud_maps folder downloaded from https://neo.gsfc.nasa.gov/view.php?datasetId=MYDAL2_M_CLD_FR into a single average cloudiness map of the world:
![merged cloudiness map of the world](consolidated_cloudmap.jpg)

4. ```sareana.py``` to combine the cloud map and the saliency map. The inverse of the cloud map is used as the blue channel of a new image and the saliency map as the green channel. The resulting output looks like this:
![combined saliency and cloudiness maps](sareana.jpg)
Additionally, ```sareana.py``` will output a region level combined slaiency and cloudiness map with equal weighting where each pixel in a region is assigned the value of the product of the saliency and inverse cloudiness in the image:
![region level world saliency](reg_sareana.jpg)
A sorted numpy array ```sorted_region_saliencys.npy``` is saved as well, containing regions and their saliencys in descending order of saliency.

5. ```landmark_analysis.py``` to use the consolidated saliency map of a region to select landmarks across a range of scales in a specific region until a fraction of the maximum saliency is reached.  
Options:
   * ```-c, --calculate```: Flag to calculate landmarks
   * ```-v, --visualize```: Flag to visualize landmarks
   * ```-s, --scales```: Scales to create landmarks for. Defaults to 5, 10, 15, 20, 25, 30, 35, 40, 45, and 50 km width squares.
   * ```-r, --region```: Region to perform landmark analysis on. Defaults to 17R.
   * ```-f, --fraction```: Fraction of maximum saliency at current scale to search landmarks down to. If the initial landmark has a saliency value of 1500 then landmark analysis would stop creating landmarks when a landmark has a saliency value less than 150 with the default ```-f``` value of 0.1
   * ```--suffix```: Suffix to add to output of landmark files. For example, ```--suffix f50``` changes ```...landmarks.npy``` to ```...landmarksf50.npy``` 
   