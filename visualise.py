import pandas as pd
import preproccess
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

df = preproccess.process_file()

X_train, X_test, Y_train, Y_test = train_test_split(df[['Pick']], df['WAV'], test_size=0.1)

poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)

ridge_reg = Ridge(alpha=100000)
ridge_reg.fit(X_train_poly, Y_train)

picks_test = X_test.copy()

X_test_poly = poly.fit_transform(X_test)

Y_pred = ridge_reg.predict(X_test_poly)

picks_test['WAV'] = Y_test
picks_test['PREDICTED_WAV'] = Y_pred


ax = picks_test.plot.scatter(x='Pick', y='WAV', s=5)
ax = picks_test.plot.scatter(x='Pick', y='PREDICTED_WAV', ax=ax, c='red', s=5)

#plt.scatter(picks_test['Pick'], picks_test['WAV'], s=5)
#plt.plot(picks_test['Pick'].values, picks_test['PREDICTED_WAV'].values)

R2_train = ridge_reg.score(X_train_poly, Y_train)
R2_test = ridge_reg.score(X_test_poly, Y_test)

print("R2 train:", R2_train)
print("R2 test:", R2_test)

plt.show()