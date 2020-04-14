**Description of the videos dataset**

The dataset is structured as follows:

* Stimuli: 19 omnidirectional videos of 20 seconds in equi-rectangular format.
* H: Folder containing the saliency maps and scanpaths from head-only movements.
* HE: Folder containing the saliency maps from head and eye movements. Scanpaths will be provided soon.
* Tools: Python scripts to parse the saliency-map binary files, and to compute saliency and scanpanth measures.
 
The details about the saliency map files and the scanpath files are:

* Saliency maps from head-only movements: Binary files representing the saliency-map sequences are provided. These sequences contain one saliency map per frame with a resolution of 2048x1024. In a binary file, the saliency values (float32) are organized row-wise and one frame after the other. For each sampled head position, the center of the viewport is considered. Then, an isotropic 3.34-degree Gaussian foveation filter centered in the view-port is applied.

* Scanpaths from head-only movements: Text files are provided with scanpaths from head movement with 100 samples per observer. Each line contains a vector that indicates the fixation index, longitude, latitude and fixation timestamp, respectively. The fixation index is incremented serially for a particular observer and resets to 0 when we reach the next observer, after all of the fixations of the given observer are reported. The fixation starting time is indicated in seconds, and latitude and longitude positions are normalized between 0 and 1 (so they should be multiplied  according to the resolution of the desired  equi-rectangular image output dimension).

* Saliency maps from head and eye movements: Binary files representing the saliency-map sequences. These sequences contain one saliency map per frame with a resolution of 2048x1024. In a binary file, the saliency values (float32) are organized row-wise and one frame after the other. For each eye fixation, an isotropic 2-degree gaussian foveation filter centered at the fixation position is applied. This process is applied to the fixations from both left and right eyes, and then combined in the final saliency map.

* Scanpaths from head and eye movements: Text files are provided with the scanpaths from both left and right eyes. Each line contains a vector that indicates the fixation index, longitude, latitude and fixation timestamp, duration, start frame and end frame, respectively. The fixation index is incremented serially for a particular observer and resets to 0 when we reach the next observer, after all of the fixations of the given observer are reported. The fixation starting time is indicated in seconds, and latitude and longitude positions are normalized between 0 and 1 (so they should be multiplied  according to the resolution of the desired  equi-rectangular image output dimension).

**Experiment**

360-degree videos were displayed in a VR headset (HTC VIVE) equipped with an SMI eye-tracker. The HTC VIVE headset allows sampling of scenes by approximately 110-degrees horizontal by 110-degrees vertical field of view (1080x1200 pixels per eye) monocularly at 90 frames per second. The eye-tracker samples gaze data at 250Hz with a precision of 0.2 degrees. A custom Unity3D scene was created to display videos. 

57 participants were recruited (25 women; age 19 to 44, mean: 25.7 years), normal or corrected-to-normal vision was verified and dominant eye of all observers was checked. All 19 videos were observed by all observers for their entire duration (20 seconds).

Observers were told to freely explore 360-degrees videos as naturally as possible while wearing a VR headset. Videos were played without audio. In order to let participants safely explore the full 360-degrees field of view, we chose to have them seat in a rolling chair. 

Participants started exploring omnidirectional contents either from an implicit longitudinal center (0-degrees and center of the equirectangular projection) or from the opposite longitude (180-degrees). Videos were observed in both rotation modalities by at least 28 participants each. We controlled observers starting longitudinal position in the scene by offsetting the content longitudinal position at stimuli onset, making sure participants started exploring 360-degrees scenes at exactly 0-degrees, or 180-degrees of longitude according to the modality. Video order and starting position modalities were cross-randomized for all participants.

Observers started the experimentation by an eye-tracker calibration, repeated every 5 videos to make sure that eye-tracker's accuracy does not degrade. the total duration of the test was less than 20 minutes.

**Citing the Database**

Please cite the following paper in your publications making use of the Salient360! database:

* Erwan David, Jesús Gutiérrez, Patrick Le Callet, Antoine Coutrot, Matthieu Perreira Da Silva, "A Dataset of Head and Eye Movements for 360° Videos", In Proceedings of the 9th ACM on Multimedia Systems Conference (MMSys'18), Amsterdam, Netherlands, June 2018.