# Create a virtual environment
python3 -m venv inceptionv3_classifier_venv

# Activate the virtual environment
source ./inceptionv3_classifier_venv/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt

# Run the application
python3 ./src/main.py

# deactivate
deactivate

# To remove the virtual environment run the following command in the terminal
#rm -rf inceptionv3_classifier_venv