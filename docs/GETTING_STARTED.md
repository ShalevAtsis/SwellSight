# Getting Started with SwellSight

Welcome to SwellSight! This guide will help you get up and running with wave analysis in just a few minutes.

## 🎯 What You'll Learn

- How to install SwellSight
- How to analyze your first beach cam image
- How to interpret the results
- Where to go next

## ⚠️ Important Notice

**This guide assumes you already have a trained model.**

If you haven't trained a model yet, please follow the **[Training from Scratch Guide](TRAINING_FROM_SCRATCH.md)** first. Training takes 2-3 days and requires:
- Beach cam images
- GPU for training
- Following the complete training pipeline

Once you have a trained model, come back here to learn how to use it!

---

## ⚡ 5-Minute Quick Start

### Step 1: Install SwellSight (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/SwellSight_Colab.git
cd SwellSight_Colab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt
```

### Step 2: Analyze Your First Image (1 minute)

```python
from src.swellsight.core.pipeline import WaveAnalysisPipeline
import cv2

# Load beach cam image
image = cv2.imread('your_beach_cam.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Analyze waves
pipeline = WaveAnalysisPipeline()
result = pipeline.process_beach_cam_image(image)

# View results
print(f"Wave Height: {result.wave_metrics.height_meters:.1f}m")
print(f"Direction: {result.wave_metrics.direction}")
print(f"Breaking Type: {result.wave_metrics.breaking_type}")
```

### Step 3: Understand the Results (2 minutes)

Your results include:

- **Wave Height**: Precise measurement in meters and feet (±0.2m accuracy)
- **Direction**: LEFT, RIGHT, or STRAIGHT (relative to beach)
- **Breaking Type**: SPILLING, PLUNGING, or SURGING
- **Confidence Scores**: Reliability of each prediction (0-100%)

Example output:
```
Wave Height: 2.3m
Direction: LEFT
Breaking Type: SPILLING
Confidence: 87.5%
```

## 📚 Next Steps

### For Beginners

1. **Try the Example Script**
   ```bash
   python examples/analyze_beach_cam.py your_image.jpg
   ```

2. **Read the User Guide**
   - [Complete User Guide](USER_GUIDE.md) - Detailed instructions
   - [Quick Reference](QUICK_REFERENCE.md) - Command cheat sheet

3. **Explore Jupyter Notebooks**
   - Start with `01_Setup_and_Installation.ipynb`
   - Follow the numbered sequence for full pipeline

### For Developers

1. **Understand the Architecture**
   - Read the [README](../README.md) for system overview
   - Review the three-stage pipeline design

2. **Explore the API**
   - Start the REST API server
   - Test endpoints with curl or Postman
   - See [API Documentation](api.md)

3. **Customize Configuration**
   - Adjust model sizes and precision
   - Set quality thresholds
   - Configure performance settings

### For Production Use

1. **Deploy the API**
   - See [Deployment Guide](deployment.md)
   - Configure monitoring and alerting
   - Set up health checks

2. **Optimize Performance**
   - Enable GPU acceleration
   - Use FP16 precision
   - Implement batch processing

3. **Monitor System Health**
   - Use built-in monitoring tools
   - Set up alerting
   - Track performance metrics

## 🎓 Learning Path

### Beginner Path (1-2 hours)
1. ✅ Install SwellSight
2. ✅ Run example script
3. ✅ Analyze a few images
4. ✅ Read Quick Reference
5. ✅ Try different configurations

### Intermediate Path (1 day)
1. ✅ Complete beginner path
2. ✅ Read full User Guide
3. ✅ Explore Jupyter notebooks
4. ✅ Try batch processing
5. ✅ Start REST API server
6. ✅ Integrate with your application

### Advanced Path (1 week)
1. ✅ Complete intermediate path
2. ✅ Study the architecture
3. ✅ Train custom models
4. ✅ Deploy to production
5. ✅ Implement monitoring
6. ✅ Optimize for your use case

## 🔧 Common First-Time Issues

### Issue: "ModuleNotFoundError"

**Solution**: Make sure you're in the project directory and virtual environment is activated
```bash
cd SwellSight_Colab
source .venv/bin/activate
pip install -r requirements/base.txt
```

### Issue: "CUDA out of memory"

**Solution**: Use CPU or smaller model
```python
config = PipelineConfig(use_gpu=False)
# or
config = PipelineConfig(depth_model_size="base")
```

### Issue: "Low confidence scores"

**Solution**: Check image quality and lighting
- Ensure ocean is visible in frame
- Use images with good lighting
- Avoid heavily compressed images

### Issue: "Slow processing"

**Solution**: Enable GPU and optimizations
```python
config = PipelineConfig(
    use_gpu=True,
    enable_optimization=True,
    depth_precision="fp16"
)
```

## 💡 Tips for Best Results

1. **Image Quality**
   - Use resolution between 480p and 4K
   - Ensure ocean is clearly visible
   - Prefer well-lit images

2. **Performance**
   - Use GPU when available (5-10x faster)
   - Enable FP16 precision (2x faster)
   - Process in batches for better throughput

3. **Reliability**
   - Check confidence scores
   - Review warnings
   - Monitor system health

4. **Production**
   - Use REST API for remote access
   - Enable monitoring and alerting
   - Implement result caching

## 📖 Documentation Index

- **[User Guide](USER_GUIDE.md)** - Complete usage instructions
- **[Quick Reference](QUICK_REFERENCE.md)** - Command cheat sheet
- **[API Documentation](api.md)** - REST API reference
- **[Training Guide](training.md)** - Model training instructions
- **[Deployment Guide](deployment.md)** - Production deployment
- **[Integration Guide](../INTEGRATION_GUIDE.md)** - System integration

## 🤝 Getting Help

### Documentation
- Check the [User Guide](USER_GUIDE.md) for detailed instructions
- Review [Quick Reference](QUICK_REFERENCE.md) for common commands
- Read the [FAQ](FAQ.md) for common questions

### Community
- [GitHub Issues](https://github.com/yourusername/SwellSight_Colab/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/yourusername/SwellSight_Colab/discussions) - Questions and discussions
- Discord Server - Real-time community support

### Support
- Email: support@swellsight.ai
- Documentation: https://docs.swellsight.ai
- Examples: Check `examples/` directory

## 🎉 You're Ready!

You now have everything you need to start analyzing waves with SwellSight. Here's what to do next:

1. **Try it out**: Run the example script with your own beach cam images
2. **Explore**: Check out the Jupyter notebooks for deeper understanding
3. **Customize**: Adjust configuration for your specific needs
4. **Share**: Join the community and share your results

Happy wave analyzing! 🏄‍♂️🌊

---

**Need more help?** Check out the [User Guide](USER_GUIDE.md) or ask in [GitHub Discussions](https://github.com/yourusername/SwellSight_Colab/discussions).
