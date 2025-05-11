# Archivo: som/config.py

# Configuración general para el SOM
SEED = 0
EPOCHS = 50
LEARNING_RATE = 0.1
RADIUS_SQ = 1.0
LR_DECAY = 0.05
RADIUS_DECAY = 0.05
BATCH_SIZE = 16
TOTAL_NODES = 100
UPDATE_METHOD = "batchmap" # opciones: 'online', 'minibatch', 'batchmap'
INIT_METHOD="random"
STEP = 3
SAVE_HISTORY = False
PROG_BAR=True
BATCH_SIZE=1000
