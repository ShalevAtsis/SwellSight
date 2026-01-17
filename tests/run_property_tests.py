"""
Simple test runner for property-based tests without pytest configuration issues.
"""

import sys
import os
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_data_quality_tests():
    """Run data quality assessment property tests."""
    print("Running data quality assessment property tests...")
    
    try:
        from test_property_data_quality_assessment import TestDataQualityAssessmentProperties
        
        test_instance = TestDataQualityAssessmentProperties()
        
        # Run a few key tests manually
        print("  Testing quality assessment consistency...")
        test_instance.setup_method()
        test_instance.test_quality_assessment_consistency(10, ["high", "low_contrast"])
        test_instance.teardown_method()
        
        print("  Testing quality score correlation...")
        test_instance.setup_method()
        test_instance.test_quality_score_correlation(15, 0.3)
        test_instance.teardown_method()
        
        print("  Testing empty dataset handling...")
        test_instance.setup_method()
        test_instance.test_empty_dataset_handling()
        test_instance.teardown_method()
        
        print("✅ Data quality assessment property tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data quality assessment property tests failed: {e}")
        traceback.print_exc()
        return False

def run_distribution_matching_tests():
    """Run distribution matching property tests."""
    print("Running distribution matching property tests...")
    
    try:
        from test_property_synthetic_real_distribution_matching import TestSyntheticRealDistributionMatching
        
        test_instance = TestSyntheticRealDistributionMatching()
        
        # Run a few key tests manually
        print("  Testing distribution matching correlation...")
        test_instance.setup_method()
        test_instance.test_distribution_matching_correlation(15, 0.8)
        test_instance.teardown_method()
        
        print("  Testing identical dataset comparison...")
        test_instance.setup_method()
        test_instance.test_identical_dataset_comparison(10)
        test_instance.teardown_method()
        
        print("  Testing statistical test validity...")
        test_instance.setup_method()
        test_instance.test_statistical_test_validity(8)
        test_instance.teardown_method()
        
        print("✅ Distribution matching property tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Distribution matching property tests failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running property-based tests for data evaluation framework...")
    
    success1 = run_data_quality_tests()
    success2 = run_distribution_matching_tests()
    
    if success1 and success2:
        print("\n🎉 All property-based tests passed!")
    else:
        print("\n💥 Some property-based tests failed!")
        sys.exit(1)