# Caltech EE 148 Final Project: "Dual Task Learning for Species and Attributes Classification"


## Abstract
Automated fine-grain species classification and physical attributes identification are important and challenging problems in ecology. State-of-the-art methods often require large amounts of image labels and natural-language captioning, both of which are labor-intensive and time-consuming tasks. In order to reduce the required amount of labelled data for training neural network-based identification and attribute classification models, we introduce a jointly-optimized model that leverages mutual information between species class and physical features. Our model simultaneously predicts both the species class and a vector of detailed image attributes from an input image. Instead of full natural-language image captions, our approach utilizes binary feature vectors that are annotated to indicate the presence or non-presence of expert pre-defined physical features. This choice of input reduces data-collection labor and time, while still providing valuable details. We test our method on the Caltech-UC San Diego Birds 200-2011 dataset and our key finding is that a dual-task approach improves multi-attribute image labelling F1-score by over 11\%. As such, we show that our method is capable of providing more accurate fine-grain details of birds than a traditional single-task model, which is useful for ecological research and educational applications. 

***Keywords: *** Computer Vision, Machine Learning, Image Classification, Ecology*

## 1. Introduction
Deep learning models have been proven to be extremely useful in recent years for an array of computer vision problems, including fine-grain species classification and image captioning. In particular, these tasks are valuable in ecological research for surveillance and conservation efforts. However, training deep-learning models often requires large amounts of labelled data. Although there is an abundance of unlabelled ecological data thanks to camera surveillance, citizen science applications, and lidar technology, labelling images is a labor-intensive and expensive task. As such, a persistent challenge in computer vision and ecology is training deep-learning models to classify species and provide fine-grain physical details of animals in images without requiring large amounts of labelled and captioned images.

Two state-of-the-art approaches to this challenge are transfer learning and multi-task classification. Transfer learning is a machine learning technique that utilizes neural network layers already trained for one task as a starting point for training another neural network for a different, but similar, task. Transfer learning has been shown to successfully reduce the amount of training samples needed to achieve high accuracy  [Pan and Yang, 2010]. Another approach is to combine input data from multiple domains, such as visual data and natural-language image captions[Gebru et al., 2017] [He and Peng, 2017].

In this paper, we propose a blended transfer learning and multi-task learning model. Our **dual-task learning model** leverages mutual information between species class and physical attributes to simultaneously predict both the species class and a vector of fine-grain attributes of an image. Our intuition is that shared information in the  dual-task approach will reduce the required amount of labelled training data required to achieve high accuracy predictions for both tasks. We test our approach using the Caltech UC-San Diego Birds (CUB) 200-2011 dataset. We implement and train two single-task models, one for species classification and one for attributes classification. Then, we implement a dual-task model for both species and attribute classification. We show that the dual-task model significantly out-performs the single-task model for multi-label attribute classification, even with fewer labelled training examples.

In the next section, we discuss previous work in transfer learning and multi-task image classification. Then, in section 3 we describe the CUB 200-2011 dataset. In section 4, we outline our single- and dual-task model architectures. In section 5, we discuss our evaluation metrics and experimental results. Finally, in sections 5 and 6, we discuss our findings and conclusions.

## 2. Related Work

### 2.1. Transfer Learning for Species Classification

A popular approach to species classification transfers inner layers from a pre-trained convolutional neural network(CNN) trained on one classification task as black-box feature extractor for training a new classification network fora different task [Pan and Yang, 2010].  Transfer learning approaches using networks trained on large-scale datasetslike ImageNet have been shown to be particularly successful for species classification, particularly the iNaturalistspecies classification challenge [Cui et al., 2018]. However, this approach does not leverage expert domain knowledgeabout which features are useful for differentiating and identifying species, risking learning coincidental correlations inthe dataset.  One existing approach to this explainability problem, shown in the work by Korsch et al. [2019], usesback-propagation to estimate relative importance of different parts of the extracted feature map for classification.

Our  model  weakly  supervises  the  visual  features  used  for  species  classification  by  fine-tuning  the  transferredfeature-extractor layers for both species and attributes classification. Since the attributes are pre-defined by experts,this approach injects domain-knowledge into the model. Additionally, by outputting attributes along with class, ourmethod provides additional information about the species classified in the image, which is useful for validating thereasonableness of the species prediction and for educational purposes.

### 2.2. Visual and Textual Inputs for Image Classification

Another approach to species classification takes advantage of different input types.  For instance, previous work inHe and Peng [2017] explores the benefits of training bird species classifiers using both visual and natural languageinputs. By fine-tuning an encoder using the CUB 200-2011 dataset to extract features and object region, they showimprovements in object localization and classification accuracy. Similarly, Gebru et al. [2017] show the benefits ofincorporating annotations along with visual data for fine-grain classification models in order to reduce the requiredamount of labelled images required for training. However, a challenge to these approaches is that natural languagecaptions are often labor-intensive and time-consuming to collect.   Moreover,  captions gathered from non-expertannotators are not guaranteed to contain useful discriminating and identifying information about the species in the image.

We similarly explore visual and non-visual training data for classification, but take an alternative approach by trainingthe model to use only visual input to predict non-visual outputs, namely a species class and a multi-label binary attributevector.  Additionally, our method does not require expert natural-language captioning, but rather multi-label binaryattribute labelling, which is easier and faster.  Another advantage to pre-defined attributes is that by specifying theattributes that annotators should label, we inject expert supervision into the data-gathering process and limit the scopeto only the most important and relevant features.

## 3. Dataset - CUB 200-2011

The Caltech-UC San Diego Birds (CUB) 200-2011 dataset contains 11,788 labeled images of birds in 200 speciescategories  [Wah  et  al.,  2011].   For  each  image,  the  dataset  includes  a  312-element  vector  of  confidence  scorescorresponding to 312 fine-grain attributes, such as bill shape, wing color, eye color, etc. We provide a visualizationof the dataset, species labels, and attribute vectors in Figure 1. These annotations are collected through the AmazonMechanical Turk platform, where the annotators are required to mark at least 10 feature labels and were are notprovided additional information about the image species. Given that annotators were not domain experts and there aresimilarities between the pre-defined attribute labels, the ground-truth annotations are subject to some inaccuracies andambiguity. To speed up training for the purposes of this paper, we train using the first 2000 images (36 species classes)of CUB 200-2011.

## 4. Methods

We propose a model that classifies bird species and bird attributes simultaneously. The intuition behind our model isthat deep image features extracted from a pre-trained model correspond to certain visual attributes, like color, size,or shape, and that similar groups of deep image features correlate to both the overall image class and its annotatedattributes. Additionally, species class information provides prior knowledge about which attributes are more likely. Forexample, an image with the species class "indigo bunting" is more likely to have blue wings and a small size.

### 4.1. Binary Multi-label Image Attributes

For  each  image  in  the  CUB  200-2011  dataset,  we  construct  a  312-element  binary  vector  where  each  elementcorresponds to one of 312 fine-grain physical attributes. For a given image, attributes that are labelled “present" by ahuman annotator are marked with a “1" in the attribute vector and other attributes are marked with a “0". By using pre-defined attribute labels specified by domain experts, as opposed to free-form natural-language captions,we inject domain knowledge into model training. Attribute vectors also allow us to crowd-source only the most relevantimage attributes, even from non-expert annotators, while avoiding irrelevant or non-discriminating information.

### 4.2. Single-Task Models

In order to contrast our proposed dual-task approach, we implement state-of-the-art models for species classificationand attribute multi-label classification separately. Both models use Google’s InceptionV3 CNN pre-trained on ImageNetas a feature extractor and are fine-tuned using CUB 200-2011 [Szegedy et al., 2015][Deng et al., 2009].

The species classifier’s decoder consists of three fully-connected layers connected to the final prediction layer of 36nodes corresponding to each species class. The species classifier is trained with a categorical-crossentropy loss functionbetween the true and predicted species label and Adam optimizer for 20 epochs with a learning rate of0.0001.

The multi-label attributes classifier’s decoder consists of two fully-connected layers connected to the final predictionlayer of 312 nodes corresponding to each attribute label. We set the confidence threshold to 0.2, above which labels areset to “1" and below which they are set to “0". The attributes classifier is trained with binary-crossentropy loss betweenthe true and predicted attribute vectors for 30 epochs with a learning rate that shrinks from 0.01 to 0.0001

### 4.3. Dual-Task Model 

Our dual species and attribute-classifier model learns to predict both species class and multi-label attribute vectorsimultaneously.  It uses the InceptionV3 CNN pre-trained on ImageNet as a shared feature extractor on the inputimage [Szegedy et al., 2015][Deng et al., 2009]. We first jointly model the extracted features for both outputs witha pooling and two fully-connected dense layers.  In these shared layers, mutual information between species classand image attributes can be shared and combined. Then, we separately train two branches, one for outputting speciesclass and one for outputting attributes class. In order to provide a fair comparison with the single-task models, bothbranches including the feature-extractor shared layers are identical to their respective single-task architectures. The fullarchitecture can be seen in Fig. 2.

We use two different loss functions for each output, corresponding to the loss functions used by the single-task models.Namely, we use categorical-cross-entropy loss for the species class output and binary-cross-entropy loss for the attributesoutput. We train the dual-task model for 30 epochs with a learning rate that shrinks from 0.01 to 0.0001.

## 5. Experiments

In our experiments, we trained a single-task species classifier, a single-task attributes classifier, and a dual species- andattributes-classifier on the first 2000 CUB 200-2011 images belonging to 36 different bird species. We partition thereduced CUB 200-2011 dataset into 80% training, 10% validation, and 10% test images with attribute labels.

### 5.1. Metrics and Baseline Scores

We evaluate the species classification score based on accuracy and F1-score and we evaluate attribute classificationperformance based on F1-score only. Since the binary attribute vectors are sparse, a high accuracy score is misleadingeven when there are lots of false 0’s in the predicted array. This can be seen in Fig. 1, where a baseline accuracy scorefor a random attribute classifier is 90.5%.

Each image belongs to one of 36 species classes and corresponds to 312 binary attributes. Each 312-element binaryattribute vector contains a mean of291’s and2830’s.  For random baseline models, we define a random speciesclassifier that has probability1/36 = 0.0278of correctly classifying an image and a random attributes classifier that assigns an element to “1" with probability29/312 = 0.093. We summarize their expected accuracy and F1-scores inTable 1. We would thus expect our species classifier predictions to reach accuracy of more than0.278and our attributesclassifier predictions to reach F1-score of more than0.171in order to beat random chance.

### 5.2. Quantitative Results

We compare the results of the single-task transfer models and dual-task model trained on the full dataset in Table2.  We note that both methods significantly outperform baseline random chance scores for both classification tasks.Although the single-task species classifier outperforms the dual-task classifier for species classification by10.09%, theclassification F1-score for attribute classification by the dual-task model exceeds the single-task attributes classifier by 11.82%.

Next, we evaluated the performance of single- and dual-task models when reducing the number of expert-labelledtraining examples. The results are plotted in Fig. 4. We observe that the dual-task model continues to dominate thesingle-task attributes classifier, even with only 40% of labelled training examples available for training.

### 5.3. Qualitative Results

From our quantitative analysis, we can generalize that for the current implementation, the dual-task model is largelyan improvement on the single-task model for attribute identification, but not so for species classification. Below, wesummarize our positive and negative findings.

#### 5.3.1. Success Cases

Our main success is showing that the dual-task model largely increases the F1-score for attributes classification. In Fig.3, we show three examples comparing single-task and dual-task model attribute predictions. In each case, we show that the dual-task modelidentifies bird attributes with a greater number of true positives and fewer numbers offalse positives and false negativesthan the single-task model. In particular, we find that the dual-task modelbetterpredicts colors, relative sizes of body parts, and fine-grain shapesthan the single-task model. The dual-task modelperforms especially well for birds that exhibit more common features or very visible features, while still identifyingfiner-grain details than the single-task model.

An additional advantage of our method is that attribute vectors are easier and faster to collect than natural-languagecaptions. Moreover, as animal species contain visually distinct features, such as color, size, and pattern, attributes areeasily labelled as present or not-present even by non-expert annotators.

#### 5.3.2. Failure Cases

Compared  to  the  single-task  species  classifier,  the  dual-task  model  struggles  with  three  examples  ofincorrectfalse species predictions.  The dual-task model is an improvement on the single-task attributes classifier, but stillprone tofalse positive and false negative attribute predictions. We include several examples of failure cases in Fig. 5.

We note that the even “incorrect" attributes that are not included in the ground truth are often still reasonable.  Forexample, “has_shape::pigeon-like" is a true label, but “has_shape::perching-like" is a false label, even though “pigeon"and “perching" are likely indistinguishable to the non-expert annotator. As such, the ground truth annotations may haveinconsistencies between similar attribute labels.

Our modelstruggles with uncommon attributes, as shown in the last example of Fig. 5. Our model easily learns toidentify common bird features and colors, such as gray color and solid patterns, but more poorly for rare attributes, likegreen wings and red breast. This may also be a limitation of the selected species included in our training set.

## 6. Discussion

In our experiments, we demonstrate that dual-task learning can improve multi-label attribute classification and provideinsights into physical species properties useful for identification.

### 6.1. Applications 

We hope that by providing improved attribute labeling for species, our model can improve ecological research andeducational tools. For research, our model can be adapted to a variety of ecological domains, such as a tool to informresearchers about key physical features present in animals being observed or tracked in the wild through camerasurveillance.  For education, our model can be used to learn about identifying features in birds for those who areinterested in learning to identify birds species in the wild.

### 6.2. Limitations 

Our work faces several limitations due to time and resources. First, the model is only trained and tested using the first2000 images form CUB 200-2011 and that is in order to work within the limited EE/CNS/CS 148 course time-frameand available computing resource constraints. Second, the binary attribute vector is constructed by assigning a “0" or“1" if an attribute is labelled “not present" or “present", without taking into account confidences for these labels. As aresult, some attributes that may actually be reasonable for an image with lower confidence are labelled “not present"in the ground truth.  Additionally, some attribute labels could be indistinguishably similar to non-experts, such as“has_nape_color::red"and“has_breast_color::red", resulting in false positives and false negatives in the ground truthattribute vectors. Finally, given time constraints, we have not yet fully examined alternative model architectures thatmay improve results, particularly for species identification accuracy.

### 6.3. Future Work 

In future work, we aim to improve our model architecture to achieve improved species classification accuracy, inaddition to our already-improved attribute classification F1-score. We would also like to evaluate our feature-extractorthat is fine-tuned using the dual-model architecture as a transferred feature-extractor for a related, single-task speciesclassification task. In this way, ecologists would be able to improve species classification accuracy, compared to anout-of-box feature-extractor, without requiring additional expert-labelled images by leveraging non-expert annotationdata for the pre-trained model. Our ultimate goal is to be able to deliver more accurate species classification with thefewest possible number of expert-labelled images.

## 7. Conclusion

In this paper, we present a new dual-task, deep-learning model architecture for simultaneously classifying bird speciesand multi-label bird attributes.  We demonstrate that a dual-task architecture greatly improves multi-label attributeprediction F1-score,  with only a slight compromise to species classification accuracy.   We hope that our modelcan empower researchers and educators to build more descriptive and detailed identification and tracking tools forconservation and education.

## 8. Acknowledgements

This work is part of the final project for Professor Pietro Perona’s EE/CNS/CS 148: Selected Topics in Computer Visioncourse at Caltech. I would like to thank Professor Perona, Sara Beery, Elijah Cole, and the other TA’s for their adviceduring this course and project.

## References

 1. Sinno Jialin Pan and Qiang Yang. A survey on transfer learning.IEEE Transactions on Knowledge and DataEngineering, 22(10):1345–1359, 2010. doi:10.1109/TKDE.2009.191.
 2. Timnit Gebru, Judy Hoffman, and Li Fei-Fei. Fine-grained recognition in the wild: A multi-task domain adaptationapproach, 2017.
 3. Xiangteng He and Yuxin Peng.Fine-graind image classification via combining vision and language.2017.doi:10.1109/CVPR.2017.775.
 4. Yin Cui, Yang Song, Chen Sun, Andrew Howard, and Serge Belongie. Large scale fine-grained categorization and domain-specific transfer learning. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,pages 4109–4118, 2018. doi:10.1109/CVPR.2018.00432.
 5. Dimitri Korsch, Paul Bodesheim, and Joachim Denzler. Classification-specific parts for improving fine-grained visualcategorization. 2019. doi:10.1007/978-3-030-33676-9_5.
 6. C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technicalreport, 2011.
 7. Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inceptionarchitecture for computer vision.CoRR, abs/1512.00567, 2015. URLhttp://arxiv.org/abs/1512.00567.
 8. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical imagedatabase. In2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.
