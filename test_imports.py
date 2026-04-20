import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(r"c:\Users\lenovo\Desktop\BP-DigitLab")
sys.path.insert(0, str(project_root))

# 测试导入
try:
    from app import bootstrap, state
    print("✓ app.bootstrap 和 app.state 导入成功")
    
    from app.pages import train_page, recognition_page
    print("✓ app.pages.train_page 和 app.pages.recognition_page 导入成功")
    
    from app.services import form_service
    print("✓ app.services.form_service 导入成功")
    
    print("\n✅ 所有主要导入都成功！")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
