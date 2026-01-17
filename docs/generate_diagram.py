"""
Generate PNG from PlantUML file using the PlantUML server
"""
import plantuml
from pathlib import Path

# PlantUML source file
puml_file = Path(__file__).parent / "architecture.puml"
png_file = Path(__file__).parent / "architecture.png"

# Read the PUML content
with open(puml_file, "r", encoding="utf-8") as f:
    puml_content = f.read()

# Use PlantUML server to generate PNG
server = plantuml.PlantUML(url="http://www.plantuml.com/plantuml/png/")

# Generate and save the PNG
print(f"Generating diagram from: {puml_file}")
try:
    # Process the file directly
    result = server.processes_file(str(puml_file), outfile=str(png_file))
    if result:
        print(f"✅ Diagram saved to: {png_file}")
    else:
        print("❌ Failed to generate diagram")
except Exception as e:
    print(f"❌ Error: {e}")
    # Fallback: try with processes method
    try:
        png_data = server.processes(puml_content)
        with open(png_file, "wb") as f:
            f.write(png_data)
        print(f"✅ Diagram saved to: {png_file}")
    except Exception as e2:
        print(f"❌ Fallback also failed: {e2}")
