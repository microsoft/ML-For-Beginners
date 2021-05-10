import pickle

model = pickle.load(open("../ufo-model.pkl", "rb"))
print(model.predict([[0, 50, -40]]))
