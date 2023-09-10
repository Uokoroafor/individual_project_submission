# Want to generate data by running the following scripts

# Run all files in root folder that end in _generator.py or _dataprep.py
for f in *_generator.py; do
    echo "Running $f"
    python $f

done

for f in *_dataprep.py; do
    echo "Running $f"
    python $f


done
