# HMM - Parts of Speech
By: Davaajav (Dona) Ganzorig  

This project uses python3.

## To run in terminal:
This package requires ```numpy```:  
```pip3 install numpy```

<br/>

And in the same directory as ```main.py```:  
```chmod +x main.py```  

Running the ```main.py``` script comes with optional arguments:
* ```--v``` Verbosity = True if included, ```False``` by default
* ```--i``` Word input file, ```words.txt``` by default
* ```--d``` Toy data file, ```toy_data.txt``` by default
* ```--o``` Output file, ```output.txt``` by default

Recommended configuration:  
```python3 main.py --i words.txt --d toy_data.txt --o output.txt```

Verbosity is not recommended because of the higher I/O cost that the
HMM demands in this part. To view verbose output anyway:  
```python3 main.py --v --i words.txt --d toy_data.txt --o output.txt```

## Inputs
The ```words.txt``` and ```toy_data.txt``` are the same as what
Dr. Goldsmith specified in the assignment writeup.

## Runtime
Since Dr. Goldsmith told us to run the iteration 400 times for each local
maximum, the runtime for this part of the HMM project takes considerably
longer time.

## Output
* ```Output.txt```: Contains iterated HMM probabilities and final
average purity table

## Extra Credit
I implemented the Part 3 Extra Credit in Spanish.

The requisite files are:
* ```words_spanish.txt```: File containing 'alphabet' words and 
parts of speech in Spanish
* ```toy_data_spanish.txt```: File containing toy data in Spanish
* ```output_spanish.txt```: Output similar to ```output.txt``` but
for the Spanish version

The command used to generate the Spanish version is:

```python3 main.py --i words_spanish.txt --d toy_data_spanish.txt --o output_spanish.txt```
