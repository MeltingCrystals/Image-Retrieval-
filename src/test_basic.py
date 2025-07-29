# test_basic.py
try:
    from mi_analysis import ColorMIAnalyzer
    print("✅ MI Analysis OK")
except Exception as e:
    print(f"❌ MI Analysis Error: {e}")

try:
    from app_pipeline import EnhancedImageSearchApp
    app = EnhancedImageSearchApp()
    print("✅ App Pipeline OK")
except Exception as e:
    print(f"❌ App Pipeline Error: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib OK")
except Exception as e:
    print(f"❌ Matplotlib Error: {e}")

try:
    import tkinter as tk
    print("✅ Tkinter OK")
except Exception as e:
    print(f"❌ Tkinter Error: {e}")

print("\nIf all ✅, you can run main.py!")
print("Run: python main.py")