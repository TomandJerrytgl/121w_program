from functions.lab1_step1 import extract_masses
import os

DATA_PATH = "data/input/p121_lab1_data/"

samples = {
    200: ("zp_mzp200_electrons.txt", "ee_mll175_electrons.txt"),
    300: ("zp_mzp300_electrons.txt", "ee_mll275_electrons.txt"),
    400: ("zp_mzp400_electrons.txt", "ee_mll375_electrons.txt"),
    500: ("zp_mzp500_electrons.txt", "ee_mll450_electrons.txt"),
    750: ("zp_mzp750_electrons.txt", "ee_mll650_electrons.txt"),
    1000: ("zp_mzp1000_electrons.txt", "ee_mll900_electrons.txt"),
}

# 👇 修改这里切换数据
mzp = 750

signal_file, background_file = samples[mzp]

signal_path = os.path.join(DATA_PATH, signal_file)
background_path = os.path.join(DATA_PATH, background_file)

print(f"\n=== Running analysis for {mzp} GeV ===")
print(f"Signal file: {signal_path}")
print(f"Background file: {background_path}")

# 读取数据
signal_masses, total_s, selected_s = extract_masses(signal_path)
background_masses, total_b, selected_b = extract_masses(background_path)

# 输出信息
print(f"\nSignal:")
print(f"  Total events: {total_s}")
print(f"  Selected events: {selected_s}")

print(f"\nBackground:")
print(f"  Total events: {total_b}")
print(f"  Selected events: {selected_b}")

print("\nFirst 10 signal masses:")
for m in signal_masses[:10]:
    print(f"{m:.3f} GeV")