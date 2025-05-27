# Archivo: som/config.py

# Configuración general para el SOM
SEED = 0
EPOCHS = 5
LEARNING_RATE = 0.1
SIGMA = 1.0
DECAY_RATE = 1
TOTAL_NODES = 100
UPDATE_METHOD = "online" # opciones: 'online', 'minibatch', 'batchmap'
INIT_METHOD="pca"
SAVE_HISTORY = False
PROG_BAR=True
BATCH_SIZE=100
