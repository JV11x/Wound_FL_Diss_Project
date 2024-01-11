# Optimising Federated Learning for Wound Recognition: A Study on Aggregation Strategies and Deployment Scenario
A joint study between AITIS Research & City, University of London

This study investigates the potential of Federated Learning (FL) for wound recognition in medical imaging, addressing the small sample size dilemma and data privacy concerns. Driven by AITIS's aim for a privacy-centric wound recognition system, the research contrasts FL strategies with traditional centralised deep learning, centring on aggregation methods and deployment contexts. The study highlighted FedProx as a top-performing FL strategy. FL methods also showed comparable performance to models trained using centralised training. The study also emphasised the environment dependent efficiency of FL techniques, with varied performance in hospital and mobile app settings. This research underscores FL's real-world applicability, championing its robustness in wound segmentation cases.

A link to the full report can be found here: https://drive.google.com/file/d/1Nv1unlF5plHJgar0dcHLCobmQka4xVLJ/view?usp=sharing 

The Project tried to address the following question:
- What is the most performant FL aggregation strategy for wound recognition?
- Can an FL approach outperform a traditional centralised deep learning scenario?
- How do models, trained using FL fare in diverse deployment scenarios?

#### To run a basic version of the experiment navigate to Test_Code in the directory and follow the instruction.

### Project Technology Stack

![image](https://github.com/JV11x/Wound_FL_Diss_Project/assets/114994769/6e7bc260-91a6-45fe-9920-6ac195b13cc3)

Complete versions of the code for each experiment are provided without in the 'Expeiment*Code' folders. However without access to GPU resources these experiments will likely be too computationally demanding for a laptop or local device to run.
To run a deperacated version of the experiments on a local machione follow the instructions in the 'Test_Code' folder.

## Experiment 1: Evaluating Different FL Aggregation Strategies
All experiments in this section employ the same U-Net image segmentation model and a consistentdataset of foot ulcers. This approach ensures a level of consistency and reproducibility across each task.

Federated learning libraries present a diverse array of aggregation strategies. The performance and effectiveness of these strategies can vary, contingent on the deployment scenario and may alter with modifications in project scale or dataset attributes. Influential factors on the performance of aggregation strategies encompass data volume, the number of possible aggregation rounds, client count, andepochs.

This experiment is designed to meticulously scrutinise the accuracy and performance of various aggregation strategies under multiple conditions. Conventional centralized model training methods, supplemented by batched approaches, will be employed as benchmarks for evaluating the FL methodologies. The outcomes aim to offer in-depth insights into the potential real-world performance of these aggregation techniques.

![image](https://github.com/JV11x/Wound_FL_Diss_Project/assets/114994769/3916e456-3f48-4700-abab-a76f2654efe6)
![image](https://github.com/JV11x/Wound_FL_Diss_Project/assets/114994769/5244a483-85eb-4510-9d38-955fd70f4d45)
![image](https://github.com/JV11x/Wound_FL_Diss_Project/assets/114994769/3a548516-c2ee-4cca-9ff3-c9231e3934da)


## Experiment 2: Comparing Centralised with Federated Learning
A paramount objective for AITIS is to establish the competitive performance of Federated Learning (FL) in contrast to the prevailing centralized machine learning training architecture. While centralized deep learning methods might fall short of FL regarding data privacy, crafting an experiment that juxtaposes centralized training with an FL approach was essential.

The experiment involves training the image segmentation model, U-Net, utilizing both conventional centralized and federated learning methods, enabling a direct comparative analysis. A uniform number of training rounds is maintained across both FL and traditional methodologies.

The delineated training scenarios encompass:
- A singular model trained to utilise the entirety of the available data.
- A model designed to assimilate new data in sequential batches following specific epochs, subsequently processing an equivalent epoch count on the newly acquired batch.
- FedAvg, applied without supplementary enhancements.
- FedAvg, complemented with an initialised model on the server.
- FedAvg, integrating a weighted aggregation strategy.
- FedAvg, incorporating both a weighted aggregation strategy and an initialised model on the server.
  
 ![image](https://github.com/JV11x/Wound_FL_Diss_Project/assets/114994769/8eec8448-d077-4fae-8a13-ed55808b81be)

## Experiment 3: Analysing Varied Deployment Contexts
This experiment is formulated to scrutinize the adaptability and effectiveness of the FL model in various deployment contexts, specifically focusing on differing client data volumes and operational environments. The investigation aims to discern the optimal application strategy, given the intrinsic constraints and capabilities of each deployment platform.
Implementation: Drawing on the insights and methodologies established in Experiments 1 and 2, Experiment 3 will delve into two distinctive deployment scenarios: a Mobile App setting and a Hospital Setting.

### Hospital Context
- In the hospital setting each client trains on 150 samples for 5 epochs, after which in trains on segments of 25 additional samples for 2 epochs, over a total of 6 aggregation rounds. This experiment assumes there are a total of 6 hospitals (clients) in total. 
![image](https://github.com/JV11x/Wound_FL_Diss_Project/assets/114994769/ac8fb31f-7a44-4d1a-b65d-bea8a670a442)

### Mobile Context
- In the mobile app setting each client trains on 50 samples for 5 epochs, after which it trains on segments of 5 additional samples for 1 epoch, over a total of 10 aggregation rounds. This experiment assumes there are a total of 15 mobile app users (clients) in total.
  
![image](https://github.com/JV11x/Wound_FL_Diss_Project/assets/114994769/359f2ac9-de35-41f1-9ab4-b582e16226ab)

