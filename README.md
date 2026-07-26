# To run the code   
1. Install the required packages using the requirements file:
   
   python -m venv .venv
   
   pip3 install -r requirements.txt
   
2. In the file data_paths.py add the corresponding local paths for the data sets.
   
   mnist, cifar10, Brain, isic, pcam.
   
3. Run the file games_local.py.
   
   You can access results in results_local/dataset_name.

4. To run the downstream tasks experiments and compute the times use the files in the folder downstrem_times
