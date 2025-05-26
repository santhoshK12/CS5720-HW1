# CS5720 Home Assignment 1

**Student:** SANTHOSH REDDY KISTIPATI 
**Course:** CS5720 Neural Networks and Deep Learning, Summer 2025

CS5720_HW1.ipynb`  
  A Google Colab notebook containing:
      Task 1: Tensor creation, reshaping, transposing, and broadcasting.  
    Task 2: Calculating and comparing loss functions, plus a bar chart.  
  Task 3 : Training a small neural network on MNIST with TensorBoard logging.

- `README.md`  
  This file with simple instructions.

---

## How to open and run

1. Go to this page
   Visit:  github link
   
2.Open the notebook in Colab 
Click on `CS5720_HW1.ipynb`.  
In the file preview, click **“Open in Colab”** (or download it and then upload into Colab).

Run every cell
In Colab, each box of code is called a “cell.”  
Click the first cell and press **Shift + Enter** (or click the ▶️ play button).  
Do this for every cell, from top to bottom.

Viewing TensorBoard (Task 3)

1. After training completes, find the code cell that starts with:
```python
%load_ext tensorboard
%tensorboard --logdir logs/fit

