# SilentSpeechClassifier
A set of research tools for silent speech classification, utilising KARA ONE database. 

## Getting Started
Download the repository, import Dataset and Classifier from SSC.py.

### Prerequisites
- mne (https://martinos.org/mne/stable/index.html)
- Scipy
- Numpy
- Sklearn
- aseegg (currently not available publicly, a wrapper for a Scipy filter)

## Known Issues
* grid search goes ape on Windows if started outside "__main__" and there is parallel computing. 
* finding best features works only with Anova and violates the assumption of normality of distribution.

## TODO
* plotting 

## Author
Wojciech Błądek

## License
Project is licensed under the MIT License.

## Acknowledgments
* Thanks to @stakar (Stanisław Karkosz) for a basis for the feature extraction module.
* Big thanks to the wonderful people behind KARA ONE (http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html)