"""
电力负荷预测系统 - 主程序
一键运行完整的预测流程
"""

import os
import sys

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator


def print_banner():
    """打印横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║          电力负荷预测系统 - 课程项目                        ║
    ║          Power Load Prediction System                    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """主函数 - 运行完整流程"""

    print_banner()

    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, 'data', 'synthetic_load.csv')
    processed_data_path = os.path.join(base_dir, 'data', 'processed_data.csv')
    featured_data_path = os.path.join(base_dir, 'data', 'featured_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')

    # 创建必要的目录
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("步骤 1/4: 数据预处理")
    print("=" * 60)

    # 检查是否已经预处理过
    if os.path.exists(processed_data_path):
        print("检测到已处理的数据，跳过预处理步骤")
    else:
        preprocessor = DataPreprocessor(raw_data_path)
        preprocessor.load_data()
        preprocessor.handle_datetime()
        preprocessor.handle_missing_values()
        preprocessor.handle_outliers()
        preprocessor.save_processed_data(processed_data_path)

    print("\n" + "=" * 60)
    print("步骤 2/4: 特征工程")
    print("=" * 60)

    # 检查是否已经完成特征工程
    if os.path.exists(featured_data_path):
        print("检测到已生成的特征数据，跳过特征工程步骤")
    else:
        engineer = FeatureEngineer(processed_data_path)
        engineer.load_data()
        feature_cols, target = engineer.prepare_data_for_modeling()
        engineer.save_featured_data(featured_data_path)

    print("\n" + "=" * 60)
    print("步骤 3/4: 模型训练")
    print("=" * 60)

    # 检查是否已经训练过模型
    if os.path.exists(os.path.join(models_dir, 'random_forest.pkl')):
        print("检测到已训练的模型，跳过训练步骤")
    else:
        trainer = ModelTrainer(featured_data_path)
        trainer.load_data()
        trainer.prepare_train_test_split()
        trainer.train_all_models()
        trainer.save_models(models_dir)

    print("\n" + "=" * 60)
    print("步骤 4/4: 模型评估")
    print("=" * 60)

    evaluator = ModelEvaluator(models_dir, featured_data_path)
    evaluator.load_models()
    evaluator.load_data()
    evaluator.evaluate_all_models()

    # 生成可视化结果
    evaluator.plot_metrics_comparison(os.path.join(results_dir, 'metrics_comparison.png'))
    evaluator.plot_predictions(os.path.join(results_dir, 'predictions.png'))
    evaluator.plot_feature_importance(os.path.join(results_dir, 'feature_importance.png'))

    # 生成报告
    evaluator.generate_report(os.path.join(results_dir, 'evaluation_report.txt'))

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n结果已保存到: {results_dir}")
    print("\n生成的文件:")
    print("  - metrics_comparison.png  : 模型性能对比图")
    print("  - predictions.png         : 预测结果可视化")
    print("  - feature_importance.png  : 特征重要性分析")
    print("  - evaluation_report.txt   : 详细评估报告")
    print("\n感谢使用电力负荷预测系统！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n错误: {str(e)}")
        import traceback

        traceback.print_exc()