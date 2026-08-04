'''
非言语行为分析统一包
封装面部 / 头部 / 躯体三大分析模块，供 Flask 应用调用
使用懒加载：仅在首次访问时导入具体分析器，避免可选依赖缺失时阻断全部模块。
'''

__all__ = ['FaceAnalyzer', 'HeadAnalyzer', 'BodyAnalyzer']


def __getattr__(name):
    """模块级懒加载：首次访问 FaceAnalyzer/HeadAnalyzer/BodyAnalyzer 时才导入。"""
    if name == 'FaceAnalyzer':
        from .face_analyzer import FaceAnalyzer
        return FaceAnalyzer
    if name == 'HeadAnalyzer':
        from .head_analyzer import HeadAnalyzer
        return HeadAnalyzer
    if name == 'BodyAnalyzer':
        from .body_analyzer import BodyAnalyzer
        return BodyAnalyzer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
