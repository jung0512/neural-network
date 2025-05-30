import sys, os 
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pylab as plt

# 오차제곱합

```
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

print(sum_squares_error(np.array(y), np.array(t)))
```
## 결과
![image](https://github.com/user-attachments/assets/75b3976b-4518-417c-8a54-59ceac2866c9)

# 교차 엔트로피 오차

```
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

print(cross_entropy_error(np.array(y), np.array(t)))
```

## 결과
![image](https://github.com/user-attachments/assets/bbdb4c29-28fd-4dfa-9b87-fc967d8b7897)

```
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)
```

## 결과
![image](https://github.com/user-attachments/assets/efb51d42-9494-42fb-be51-b6f54a9c91c3)


```
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```

# 교차 엔트로피

```
def cross_entropy_error(y, t):
    if y.npim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

# 미분

```
# 나쁜 구현 예
def numerical_diff(f, x):
    h = 1e-50
    return (f(x + h) - f(x)) / h

#좋은 구현 예
def numerical_difff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h))/ (2*h)

## 수치 미분의 예

# y = 0.01x**2 + 0.1x 구현
def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1) #0에서 20까지 0.1 간격의 배열을 x를 만든다(20은 미포함)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()

# x = 5일 때와 10일 때 
print(numerical_difff(function_1, 5))
print(numerical_difff(function_1, 10))
```
## 결과
![image](https://github.com/user-attachments/assets/36ffa1f1-947d-4e58-9798-5b1fce26cb19)
![image](https://github.com/user-attachments/assets/61807110-c59f-4aa6-a813-9613bae4f8c9)

# 편미분

```
# f(x_0, x_1) = x_0^2 + x_1^2 식 구현
def function_2(x):
    return x[0]**2 + x[1]**2
    #또는 return np.sum(x ** 2)
```
