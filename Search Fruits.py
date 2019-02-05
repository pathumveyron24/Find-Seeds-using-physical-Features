from sklearn import tree

# smooth = 1 and bumpy =0 machines use real data
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

labels = [0, 0, 1, 1]  # apple = 0 and oranges = 1

adoreclassifier = tree.DecisionTreeClassifier()

adoreclassifier = adoreclassifier.fit(features, labels)

print(adoreclassifier.predict([[150, 0]]))
