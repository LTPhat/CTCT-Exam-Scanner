# Multiple Choices Exam Scanner
- Digital Image Processing Course Project (EE3035).

## Proposed System Diagram

![alt text](https://github.com/LTPhat/CTCT-Multiple-Choices-Exam-Scanner/blob/main/assets/diagram.jpg)

## Tutorials
- Install Anaconda from this url https://www.anaconda.com/download. (Ignore this step if Anaconda is already installed).
- Create new conda environment.
  
```python
conda create -n <your_env_name> python==3.10
```
- Activate new created environment.
```python
conda activate <your_env_name>
```
- Install libraries and packages.
```python
pip install -r requirement.txt
```
- Check folder information
  - ``./samples:`` Place your input images here.
  - ``./ans_keys:`` Put your answer key (*.txt) file here. Set up answers as the below format.
    
  ```sh
  1A
  2A
  3A
  4A
  5A
  6E
  ...
  ```
  - ``./results:`` Where the scanning result is stored as ``result.csv`` file.

- Run GUI
  
```shell
python GUI.py
```

## Demo
  
![alt text](https://github.com/LTPhat/CTCT-Multiple-Choices-Exam-Scanner/blob/main/assets/gui_checked.png)


![alt text](https://github.com/LTPhat/CTCT-Multiple-Choices-Exam-Scanner/blob/main/assets/result.png)


