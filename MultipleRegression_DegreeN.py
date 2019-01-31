 
#請畫出各種degree的多項式回歸和訓練樣本數N的關係


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error #MSE計算誤差



x = np.linspace(0, 2*np.pi, 10) 
#建立一個從 0 到 2*np.pi ，共分100塊相等等距矩陣
y = np.sin(x)
plt.subplot(2,1,1)
plt.scatter(x, y)

x1 = np.linspace(0, 2*np.pi, 10)
y1= np.sin(x1)+np.random.randn(len(x1))/5.0
#輸出標準正態分佈的矩陣
plt.subplot(3,1,1)
plt.scatter(x1, y1)

x_t = np.linspace(0, 2*np.pi, 10)
y_t= np.sin(x1)+np.random.randn(len(x1))/5.0
plt.subplot(5,1,1)
plt.scatter(x_t, y_t)


#利用直線方程式擬合數據點
slr = LinearRegression()
#x1=pd.DataFrame(x1)
x1=x1.reshape(-1, 1)
#一個參數為-1時，reshape函數會根據另一參數維度計算出另一shape屬性值
slr.fit(x1, y1)
print("訓練樣本數N=10")
print("直線-迴歸係數:", slr.coef_)
print("截距:", slr.intercept_ )

X_plt = np.linspace(0, 6, 1000).reshape(-1, 1)
y_plt = np.dot(X_plt, slr.coef_.T) + slr.intercept_

predicted_y1 = slr.predict(x1)
plt.subplot(5,1,1)
plt.plot(x1, predicted_y1)


#MSE計算直線方程式誤差
h = np.dot(x1.reshape(-1, 1), slr.coef_.T) + slr.intercept_
print("殘差 ",mean_squared_error(h, y1)) 


#擬合3次方程式
poly_features_b = PolynomialFeatures(degree=3, include_bias=False)
X_poly_b = poly_features_b.fit_transform(x1)
lin_reg_b = LinearRegression()
lin_reg_b.fit(X_poly_b, y1)
print("3次方程式-迴歸係數:",  lin_reg_b.coef_)
print("截距:", lin_reg_b.intercept_ )

X_plot = np.linspace(0, 6,1000).reshape(-1, 1)
X_plot_poly = poly_features_b.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_b.coef_.T) + lin_reg_b.intercept_
plt.subplot(5,1,2)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')


#擬合6次方程式
poly_features_c = PolynomialFeatures(degree=6, include_bias=False)
X_poly_c = poly_features_c.fit_transform(x1)
lin_reg_c = LinearRegression()
lin_reg_c.fit(X_poly_c, y1)
print("6次方程式-迴歸係數:", lin_reg_c.coef_)
print("截距:", lin_reg_c.intercept_ )

X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_c.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_c.coef_.T) + lin_reg_c.intercept_
plt.subplot(5,1,3)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')



#擬合9次方程式
poly_features_d = PolynomialFeatures(degree=9, include_bias=False)
X_poly_d = poly_features_d.fit_transform(x1)
lin_reg_d = LinearRegression()
lin_reg_d.fit(X_poly_d, y1)
print("9次方程式-迴歸係數:", lin_reg_d.coef_)
print("截距:", lin_reg_d.intercept_ )

X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_d.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_d.coef_.T) + lin_reg_d.intercept_
plt.subplot(5,1,4)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')


#擬合12次方程式
poly_features_e = PolynomialFeatures(degree=12, include_bias=False)
X_poly_e = poly_features_e.fit_transform(x1)
lin_reg_e = LinearRegression()
lin_reg_e.fit(X_poly_e, y1)
print(lin_reg_e.intercept_, lin_reg_e.coef_)
print("12次方程式-迴歸係數:", lin_reg_e.coef_)
print("截距:", lin_reg_e.intercept_ )

X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_e.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_e.coef_.T) + lin_reg_e.intercept_
plt.subplot(5,1,5)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')

plt.show()















print("訓練樣本數N=50")

x = np.linspace(0, 2*np.pi, 50) 
#建立一個從 0 到 2*np.pi 
y = np.sin(x)
plt.subplot(2,1,1)
plt.scatter(x, y)

x1 = np.linspace(0, 2*np.pi, 50)
y1= np.sin(x1)+np.random.randn(len(x1))/5.0
#輸出標準正態分佈的矩陣
plt.subplot(3,1,1)
plt.scatter(x1, y1)

x_t = np.linspace(0, 2*np.pi, 50)
y_t= np.sin(x1)+np.random.randn(len(x1))/5.0
plt.subplot(5,1,1)
plt.scatter(x_t, y_t)


#利用直線方程式擬合數據點
slr = LinearRegression()
#x1=pd.DataFrame(x1)
x1=x1.reshape(-1, 1)
#一個參數為-1時，reshape函數會根據另一參數維度計算出另一shape屬性值
slr.fit(x1, y1)


X_plt = np.linspace(0, 6, 1000).reshape(-1, 1)
y_plt = np.dot(X_plt, slr.coef_.T) + slr.intercept_

predicted_y1 = slr.predict(x1)
plt.subplot(5,1,1)
plt.plot(x1, predicted_y1)




#擬合3次方程式
poly_features_b = PolynomialFeatures(degree=3, include_bias=False)
X_poly_b = poly_features_b.fit_transform(x1)
lin_reg_b = LinearRegression()
lin_reg_b.fit(X_poly_b, y1)


X_plot = np.linspace(0, 6,1000).reshape(-1, 1)
X_plot_poly = poly_features_b.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_b.coef_.T) + lin_reg_b.intercept_
plt.subplot(5,1,2)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')


#擬合6次方程式
poly_features_c = PolynomialFeatures(degree=6, include_bias=False)
X_poly_c = poly_features_c.fit_transform(x1)
lin_reg_c = LinearRegression()
lin_reg_c.fit(X_poly_c, y1)


X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_c.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_c.coef_.T) + lin_reg_c.intercept_
plt.subplot(5,1,3)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')



#擬合9次方程式
poly_features_d = PolynomialFeatures(degree=9, include_bias=False)
X_poly_d = poly_features_d.fit_transform(x1)
lin_reg_d = LinearRegression()
lin_reg_d.fit(X_poly_d, y1)


X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_d.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_d.coef_.T) + lin_reg_d.intercept_
plt.subplot(5,1,4)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')


#擬合12次方程式
poly_features_e = PolynomialFeatures(degree=12, include_bias=False)
X_poly_e = poly_features_e.fit_transform(x1)
lin_reg_e = LinearRegression()
lin_reg_e.fit(X_poly_e, y1)


X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_e.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_e.coef_.T) + lin_reg_e.intercept_
plt.subplot(5,1,5)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')




plt.show()









print("訓練樣本數N=100")

x = np.linspace(0, 2*np.pi, 100) 
#建立一個從 0 到 2*np.pi ，共分100塊相等等距矩陣
y = np.sin(x)
plt.subplot(2,1,1)
plt.scatter(x, y)

x1 = np.linspace(0, 2*np.pi, 100)
y1= np.sin(x1)+np.random.randn(len(x1))/5.0
#輸出標準正態分佈的矩陣
plt.subplot(3,1,1)
plt.scatter(x1, y1)

x_t = np.linspace(0, 2*np.pi, 100)
y_t= np.sin(x1)+np.random.randn(len(x1))/5.0
plt.subplot(5,1,1)
plt.scatter(x_t, y_t)


#利用直線方程式擬合數據點
slr = LinearRegression()
#x1=pd.DataFrame(x1)
x1=x1.reshape(-1, 1)
#一個參數為-1時，reshape函數會根據另一參數維度計算出另一shape屬性值
slr.fit(x1, y1)


X_plt = np.linspace(0, 6, 1000).reshape(-1, 1)
y_plt = np.dot(X_plt, slr.coef_.T) + slr.intercept_

predicted_y1 = slr.predict(x1)
plt.subplot(5,1,1)
plt.plot(x1, predicted_y1)



#擬合3次方程式
poly_features_b = PolynomialFeatures(degree=3, include_bias=False)
X_poly_b = poly_features_b.fit_transform(x1)
lin_reg_b = LinearRegression()
lin_reg_b.fit(X_poly_b, y1)


X_plot = np.linspace(0, 6,1000).reshape(-1, 1)
X_plot_poly = poly_features_b.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_b.coef_.T) + lin_reg_b.intercept_
plt.subplot(5,1,2)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')


#擬合6次方程式
poly_features_c = PolynomialFeatures(degree=6, include_bias=False)
X_poly_c = poly_features_c.fit_transform(x1)
lin_reg_c = LinearRegression()
lin_reg_c.fit(X_poly_c, y1)


X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_c.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_c.coef_.T) + lin_reg_c.intercept_
plt.subplot(5,1,3)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')



#擬合9次方程式
poly_features_d = PolynomialFeatures(degree=9, include_bias=False)
X_poly_d = poly_features_d.fit_transform(x1)
lin_reg_d = LinearRegression()
lin_reg_d.fit(X_poly_d, y1)


X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_d.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_d.coef_.T) + lin_reg_d.intercept_
plt.subplot(5,1,4)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')


#擬合12次方程式
poly_features_e = PolynomialFeatures(degree=12, include_bias=False)
X_poly_e = poly_features_e.fit_transform(x1)
lin_reg_e = LinearRegression()
lin_reg_e.fit(X_poly_e, y1)


X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
X_plot_poly = poly_features_e.fit_transform(X_plot)
y_plot = np.dot(X_plot_poly, lin_reg_e.coef_.T) + lin_reg_e.intercept_
plt.subplot(5,1,5)
plt.plot(X_plot, y_plot, 'r-')
plt.plot(x1, y1, 'b.')




plt.show()
