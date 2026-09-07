with open(r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work\tests\unit\test_audio_analysis.py', 'rb') as f:
    content = f.read()

# Fix the for loop indentation - it should be at 12 spaces (4 more than method body at 8 spaces)
old = b'        for k, v in result.features.items():\n            if k in ("rms_energy_db", "energy_mean"):\n                assert v == -120.0, f"Expected -120.0 for {k}, got {v}"\n            else:\n                assert v == 0.0, f"Expected 0.0 for {k}, got {v}"\n            \n            continue\n        \n    def test_extract_sine_wave'
new = b'            for k, v in result.features.items():\n                if k in ("rms_energy_db", "energy_mean"):\n                    assert v == -120.0, f"Expected -120.0 for {k}, got {v}"\n                else:\n                    assert v == 0.0, f"Expected 0.0 for {k}, got {v}"\n                \n                continue\n            \n    def test_extract_sine_wave'

content = content.replace(old, new)

with open(r'C:\Users\theat\OneDrive\Desktop\BE_Project\my_app\app_work\tests\unit\test_audio_analysis.py', 'wb') as f:
    f.write(content)
print('Fixed')