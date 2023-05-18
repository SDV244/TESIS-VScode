"""
Created on Sun Aug  7 14:50:41 2022

@author: Michael || Sebastian

Inception V3 Model Training Script

This script trains a Inception V3 model for multilabel classification using the given dataset. It includes options for single-layer and two-layer models, as well as K-fold cross-validation.


"""

# INCEPTION V3 MODELS TRAINING SCRIPT
#%%
#LIBRARIES

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


#END OF LIBRARIES



#%%

#1Layer



# Load the pre-trained InceptionV3 model
base_model = InceptionV3(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a custom output layer for multilabel classification
x = Flatten()(base_model.output)
x = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)
x = keras.layers.Dropout(0.5)(x)
# x = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
# x = keras.layers.Dropout(0.5)(x)
output = Dense(3, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=[Precision(), Recall(), AUC(curve='ROC'), AUC(curve='PR', name='PR AUC'), 'binary_accuracy']
)

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('/content/drive/MyDrive/SUPER_AUG_MODELS/INCEPTION_V3/NOAUG_InceptionV3_1LYR_RegL2_Lr_00001.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model for 100 epochs with batch size 32
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=1000,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose = 1
)



# Plot training & validation PRAUC values
plt.plot(history.history['PR AUC'])
plt.plot(history.history['val_PR AUC'])
plt.title('Model PR AUC')
plt.ylabel('PR AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig(f'/content/drive/MyDrive/SUPER_AUG_MODELS/INCEPTION_V3/PLOT_PRAUC_NOAUG_InceptionV3_1LYR_RegL2_Lr_00001.h5.png')



# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'/content/drive/MyDrive/SUPER_AUG_MODELS/INCEPTION_V3/PLOT_LOSS_NOAUG_InceptionV3_1LYR_RegL2_Lr_00001.h5.png')

# Evaluate the model on the test set using F1 score
y_pred = model.predict(X_test)
test_f1_score = f1_score(y_test, y_pred > 0.5, average=None)
test_precision = Precision()(y_test, y_pred).numpy()
test_recall = Recall()(y_test, y_pred).numpy()
test_roc_auc = AUC(curve='ROC')(y_test, y_pred).numpy()
test_pr_auc = average_precision_score(y_test, y_pred, average='micro')
print(f'Test F1 score: {test_f1_score}')
print(f'Test precision: {test_precision}')
print(f'Test recall: {test_recall}')
print(f'Test ROC AUC: {test_roc_auc}')
print(f'Test PR AUC: {test_pr_auc}')


#%%


#2LAYERS



# Load the pre-trained InceptionV3 model
base_model = InceptionV3(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a custom output layer for multilabel classification
x = Flatten()(base_model.output)
x = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)
x = keras.layers.Dropout(0.5)(x)
x = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
x = keras.layers.Dropout(0.5)(x)
output = Dense(3, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #)(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=[Precision(), Recall(), AUC(curve='ROC'), AUC(curve='PR', name='PR AUC'), 'binary_accuracy']
)

# Set up early stopping and model checkpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
checkpoint = ModelCheckpoint('/content/drive/MyDrive/Models/SUPER_AUG_MODELS/INCEPTION_V3/SUPER_AUG_InceptionV3_2LYR.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model for 100 epochs with batch size 32
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=1000,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose = 1
)




# Plot training & validation PRAUC values
plt.plot(history.history['PR AUC'])
plt.plot(history.history['val_PR AUC'])
plt.title('Model PR AUC')
plt.ylabel('PR AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig(f'/content/drive/MyDrive/Models/SUPER_AUG_MODELS/INCEPTION_V3/PLOT_PRAUC_SUPER_AUG_InceptionV3_2LYR.png')



# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig(f'/content/drive/MyDrive/Models/SUPER_AUG_MODELS/INCEPTION_V3/PLOT_LOSS_SUPER_AUG_InceptionV3_2LYR.png')


# Evaluate the model on the test set using F1 score
y_pred = model.predict(X_test)
test_f1_score = f1_score(y_test, y_pred > 0.5, average=None)
test_precision = Precision()(y_test, y_pred).numpy()
test_recall = Recall()(y_test, y_pred).numpy()
test_roc_auc = AUC(curve='ROC')(y_test, y_pred).numpy()
test_pr_auc = average_precision_score(y_test, y_pred, average='micro')
print(f'Test F1 score: {test_f1_score}')
print(f'Test precision: {test_precision}')
print(f'Test recall: {test_recall}')
print(f'Test ROC AUC: {test_roc_auc}')
print(f'Test PR AUC: {test_pr_auc}')



#%%

#1 LAYER KFOLD
# import necessary libraries


# Define the number of folds
n_splits = 2

# Initialize the KFold object
kf = KFold(n_splits=n_splits)

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Split the data into train and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Load the pre-trained MobileNet model
    base_model = InceptionV3(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom output layer for multilabel classification
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
    x = keras.layers.Dropout(0.5)(x)
    #x = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
    #x = keras.layers.Dropout(0.5)(x)
    output = Dense(3, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) 

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=[Precision(), Recall(), AUC(curve='ROC'), AUC(curve='PR', name='PR AUC'), 'binary_accuracy']
    )

    # Set up early stopping and model checkpoint callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True,start_from_epoch=6)
    checkpoint = ModelCheckpoint(f'/content/drive/MyDrive/Models/SUPER_AUG_MODELS/INCEPTION_V3/SUPER_AUG_InceptionV3_1LYR_fold_{fold}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the model for 100 epochs with batch size 32
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=1000,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )


    


    # Plot training & validation PRAUC values
    plt.plot(history.history['PR AUC'])
    plt.plot(history.history['val_PR AUC'])
    plt.title('Model PR AUC')
    plt.ylabel('PR AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig(f'/content/drive/MyDrive/Models/SUPER_AUG_MODELS/INCEPTION_V3/PLOT_PRAUC_SUPER_AUG_InceptionV3_1LYR_fold_{fold}.png')



    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'/content/drive/MyDrive/Models/SUPER_AUG_MODELS/INCEPTION_V3/PLOT_LOSS_SUPER_AUG_InceptionV3_1LYR_fold_{fold}.png')

# Evaluate the model on the test set using F1 score
    y_pred = model.predict(X_test)
    test_f1_score = f1_score(y_test, y_pred > 0.5, average=None)
    test_precision = Precision()(y_test, y_pred).numpy()
    test_recall = Recall()(y_test, y_pred).numpy()
    test_roc_auc = AUC(curve='ROC')(y_test, y_pred).numpy()
    test_pr_auc = average_precision_score(y_test, y_pred, average='micro')
    print(f'Test F1 score: {test_f1_score}')
    print(f'Test precision: {test_precision}')
    print(f'Test recall: {test_recall}')
    print(f'Test ROC AUC: {test_roc_auc}')
    print(f'Test PR AUC: {test_pr_auc}')
    

#%%


#K-FOLD VERSION
#2 LAYER
# import necessary libraries


# Define the number of folds
n_splits = 2

# Initialize the KFold object
kf = KFold(n_splits=n_splits)

# Iterate over the folds
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    
    # Split the data into train and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Load the pre-trained MobileNet model
    base_model = InceptionV3(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom output layer for multilabel classification
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
    x = keras.layers.Dropout(0.5)(x)
    x = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) #
    x = keras.layers.Dropout(0.5)(x)
    output = Dense(3, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) 

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model with Adam optimizer, binary crossentropy loss, and metrics AUC and binary accuracy
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=[Precision(), Recall(), AUC(curve='ROC'), AUC(curve='PR', name='PR AUC'), 'binary_accuracy']
    )

    # Set up early stopping and model checkpoint callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True,start_from_epoch=6)
    checkpoint = ModelCheckpoint(f'/content/drive/MyDrive/Models/SUPER_AUG_MODELS/INCEPTION_V3/SUPER_AUG_InceptionV3_2LYR_fold_{fold}.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train the model for 100 epochs with batch size 32
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=1000,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )


    # Plot training & validation PRAUC values
    plt.plot(history.history['PR AUC'])
    plt.plot(history.history['val_PR AUC'])
    plt.title('Model PR AUC')
    plt.ylabel('PR AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig(f'/content/drive/MyDrive/Models/SUPER_AUG_MODELS/INCEPTION_V3/PLOT_PRAUC_SUPER_AUG_InceptionV3_2LYR_fold_{fold}.png')



    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'/content/drive/MyDrive/Models/SUPER_AUG_MODELS/INCEPTION_V3/PLOT_LOSS_SUPER_AUG_InceptionV3_2LYR_fold_{fold}.png')


# Evaluate the model on the test set using F1 score
    y_pred = model.predict(X_test)
    test_f1_score = f1_score(y_test, y_pred > 0.5, average=None)
    test_precision = Precision()(y_test, y_pred).numpy()
    test_recall = Recall()(y_test, y_pred).numpy()
    test_roc_auc = AUC(curve='ROC')(y_test, y_pred).numpy()
    test_pr_auc = average_precision_score(y_test, y_pred, average='micro')
    print(f'Test F1 score: {test_f1_score}')
    print(f'Test precision: {test_precision}')
    print(f'Test recall: {test_recall}')
    print(f'Test ROC AUC: {test_roc_auc}')
    print(f'Test PR AUC: {test_pr_auc}')

    
