#Get the input video
echo "Downloading input video..."
gdown "https://drive.google.com/uc?export=download&id=1czq9F8TX8bUkq086IMDL-6H2BUe-GUbn"
unzip videos.zip
rm videos.zip

#Get the VIBE model
echo "Downloading VIBE model..."
mkdir -p data
cd data
gdown "https://drive.google.com/uc?export=download&id=1r5CoNee5YDb62nLNcfhl4hFK1XY9qUTx"
unzip vibe_data.zip
rm vibe_data.zip
cd ..

echo "Running lane detection..."
python lane.py

echo "Running 3D Bounding Box pipeline..."
python run.py


python demo.py output_videos/output.mp4
echo "Process completed successfully!"
