# Deep_Coefficient_Varying_DAG_Models




# Application 2. Whole Slide Image Conditional Dependency Learning
A whole silde image (WSI), may consist of over 1 billion pixels, which is not suitable for being imported into a convolutional neural network directly. Hence, WSI data are usually cut into patches like 256 * 256. Later every single patch is embedded by a ResNet-based MLP, so the WSI is transformed into a matrix in shape n * d, where n denotes the number of patches and d is the dimension output from the MLP. 

The core challenge is that often only the label of the entire WSI data is available. In another word, we can not distinguish which exact patch correspond with the label. A WSI data is therefore callded a **bag**, and each patch inside it is **instance**. Hence, the task is referred as **multiple instance learning**. 
