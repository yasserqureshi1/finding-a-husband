# Finding a Husband: Using explainable AI to identify male mosquito flight differences

## Abstract
Mosquito-borne diseases alone account for around a million deaths annually. There is a constant need for novel intervention mechanisms to mitigate transmission, especially as current insecticidal methods become less effective with the rise of insecticide resistance among mosquito populations. Previously, we used a near infrared tracking system to describe the behaviour of mosquitoes at a human-occupied bednet, work that eventually led to an entirely novel bednet design. Advancing that approach, here we report on the use of trajectory analysis of mosquito flight, using machine learning methods. This largely unexplored application has significant potential for providing useful insights into the behaviour of mosquitoes and other insects. In this work, a novel methodology applies anomaly detection to distinguish male mosquito tracks from females and couples. The proposed pipeline uses new feature engineering techniques and splits each track into segments to avoid data leaks. Each segment is classified individually, and the outcomes are combined to classify whole tracks. By interpreting the model using SHAP values, features of flight that contribute to the differences between sexes are found and are explained by expert opinion. This methodology was tested using 3D tracks generated from mosquito mating swarms in the field and obtained a balanced accuracy of 64.5% and a ROC AUC score of 68.4%. Such a system can be used in a wide variety of trajectory domains to detect and analyse the behaviours of different classes e.g., sex, strain, species. The results of this study can support genetic mosquito control interventions for which mating represents a key event for their success.

## Getting Started

To get started, you will need Python 3.8+ and Jupyter Notebook. You can clone the repository using the command:
```
git clone https://github.com/yasserqureshi1/finding-a-husband.git
```

To install the required dependencies use the command:
```
pip install -r requirements.txt
```


## Authors

Yasser M. Qureshi, Voloshin Voloshin, Luca Facchinelli, Philip J. McCall, Olga Chervova, Cathy E. Towers, James A. Covington and David P. Towers

## License

Distributed under the BSD-3 Clause license