# Implementation Guide for Scotiabank

**Document Purpose:** Step-by-step guide for Scotiabank's technical team to implement and operationalize the sentiment analysis tool.

**Prepared for:** Scotiabank IT and Corporate Strategy Teams  
**Prepared by:** MGSC 661 Consulting Team  
**Date:** December 2025

## Implementation Timeline

**Week 1: Setup and Testing**
- Install Python environment and dependencies
- Download required data files (dictionary, stopwords)
- Test with sample transcripts
- Validate output quality

**Week 2: Integration**
- Configure bank URLs for Q1 2025 transcripts
- Run full analysis on all Big 5 banks
- Train key users on interpretation
- Document internal processes

**Week 3+: Operationalization**
- Schedule quarterly analysis runs
- Establish reporting workflow
- Integrate into existing strategy reports

## Technical Setup

### Environment Setup

**Option A: Local Installation (Recommended for initial testing)**

1. Install Python 3.8+ from python.org
2. Install Jupyter Notebook: `pip install jupyter`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `jupyter notebook multi_bank_sentiment_streamlined.ipynb`

**Option B: Cloud Environment (Recommended for production)**

Consider deploying on:
- Google Colab (free, no installation required)
- AWS SageMaker (enterprise-grade, scalable)
- Azure ML Studio (if using Microsoft ecosystem)

**Option C: Scotiabank Server Deployment**

Work with IT to deploy on internal analytics server with:
- Python 3.8+ installed
- 16GB RAM minimum
- Access to internet for PDF downloads
- Scheduled job capability (cron/Task Scheduler)

### Required Data Files

**1. Loughran-McDonald Dictionary**
- Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
- File name: `Loughran-McDonald_MasterDictionary_1993-2024.csv`
- Update annually (dictionary updates in January)

**2. Stopwords List**
- Standard English stopwords
- File name: `stopwords.txt`
- Available from NLTK corpus or consulting team

**3. Bank Transcript URLs**
- Maintain list of current quarter URLs
- Update configuration file quarterly
- Test URLs for accessibility before analysis

## User Training

### Key Users to Train

1. **Corporate Strategy Analysts** (Primary users)
   - How to configure bank list
   - How to run analysis
   - How to interpret results
   - How to export for reports

2. **IT Support** (Technical administrators)
   - Environment setup and troubleshooting
   - Dependency management
   - Scheduled job configuration
   - Error handling and logging

3. **Strategy Leadership** (Report consumers)
   - Understanding methodology
   - Interpreting visualizations
   - Using results for decision-making
   - Limitations and appropriate use

### Training Materials Provided

1. This implementation guide
2. Main README with methodology details
3. Sample output and interpretation guide
4. Troubleshooting FAQ
5. Recorded walkthrough (if requested)

## Quarterly Workflow

### Week of Earnings Releases

**Day 1-2: Data Collection**
- Gather transcript URLs for all banks as they release earnings
- Verify PDFs are accessible and complete
- Update configuration file with new URLs

**Day 3: Analysis Execution**
- Run Jupyter notebook (approximately 15-20 minutes total)
- Review output for any errors or anomalies
- Validate example sentences make sense

**Day 4: Report Generation**
- Extract key findings (rankings, score changes)
- Create PowerPoint using visualizations
- Write executive summary with insights
- Compare to previous quarter results

**Day 5: Stakeholder Distribution**
- Share report with strategy leadership
- Present findings in strategy meeting
- Archive results for future reference

### Monthly Monitoring

- Check for any methodology updates from research sources
- Review new FinBERT model versions
- Update documentation as needed

### Annual Maintenance

- Update Loughran-McDonald dictionary (January release)
- Review and update stopword list if needed
- Assess if new competitor banks should be added
- Evaluate tool effectiveness and enhancement opportunities

## Integration with Existing Systems

### Recommended Integrations

**1. Document Management**
- Store quarterly results in SharePoint/OneDrive
- Maintain consistent folder structure by quarter
- Archive transcript PDFs alongside results

**2. Business Intelligence Tools**
- Import CSV results into Tableau/Power BI
- Create dashboard for historical trend tracking
- Automate quarterly dashboard updates

**3. Collaboration Platforms**
- Post analysis summaries in Microsoft Teams channel
- Create recurring strategy meeting agenda item
- Share key insights in internal newsletters

**4. Calendar and Reminders**
- Set recurring reminders for earnings week
- Schedule regular analysis runs
- Block analyst time for report generation

## Quality Assurance Checklist

Before finalizing quarterly analysis, verify:

- [ ] All transcript URLs are correct and accessible
- [ ] Processing completed without errors
- [ ] Rankings make intuitive sense given market conditions
- [ ] Example sentences appear relevant and substantive
- [ ] Both methods (Dictionary and FinBERT) completed successfully
- [ ] CSV export generated correctly
- [ ] Visualizations display properly
- [ ] Quarter-over-quarter comparison reviewed
- [ ] Any anomalies investigated and explained

## Troubleshooting Common Issues

### Issue: FinBERT model download fails

**Symptoms:** Error during Step 5, "Failed to download model"

**Solutions:**
1. Check internet connectivity
2. Verify firewall allows access to huggingface.co
3. Try manual download: `transformers-cli download ProsusAI/finbert`
4. If persistent, download model files directly and load from local path

### Issue: PDF text extraction is garbled

**Symptoms:** Example sentences contain strange characters or are incomplete

**Solutions:**
1. Verify PDF is text-based (not scanned image)
2. Try opening PDF manually to confirm readability
3. Check if bank has alternative transcript format (Word, HTML)
4. May need OCR pre-processing for scanned documents

### Issue: Processing takes extremely long (30+ minutes)

**Symptoms:** Notebook runs but takes excessive time

**Solutions:**
1. Check system RAM usage (may need more memory)
2. Reduce number of banks processed at once
3. Close other memory-intensive applications
4. Consider upgrading to cloud environment with more resources

### Issue: Scores seem unrealistic

**Symptoms:** Bank with known issues shows very positive sentiment

**Solutions:**
1. Review example sentences to understand what drove score
2. Check if transcript URL is correct (not using previous quarter)
3. Verify substantive content detection is working (not analyzing boilerplate)
4. Consider that sentiment reflects language, not underlying fundamentals

### Issue: Banks rank differently between methods

**Symptoms:** Dictionary and FinBERT produce different rankings

**Solutions:**
- This is normal and expected
- Dictionary captures word frequency
- FinBERT captures contextual meaning
- Review example sentences to understand discrepancy
- Use FinBERT ranking for strategic decisions
- Use Dictionary ranking for compliance documentation
- Report both rankings in final analysis

## Security and Data Handling

### Data Classification

**Public Data:**
- Earnings transcripts (publicly available)
- Tool source code
- Methodology documentation

**Internal Use:**
- Analysis results and rankings
- Quarterly reports and presentations
- Strategic insights derived from analysis

**Confidential:**
- Internal strategic discussions based on analysis
- Scotiabank's specific planned actions
- Competitive intelligence assessments

### Best Practices

1. Store transcripts and results on secure Scotiabank servers
2. Limit report distribution to authorized strategy personnel
3. Redact specific competitor details when sharing externally
4. Do not share proprietary analysis methodology with competitors
5. Comply with insider trading regulations (analysis is public data, but timing matters)

## Performance Optimization

### For Faster Processing

1. **Use SSD storage** for faster file I/O
2. **Close unnecessary applications** to free RAM
3. **Process in batches** if analyzing many banks
4. **Cache FinBERT model** after first download
5. **Use GPU** if available (speeds up FinBERT significantly)

### For Better Results

1. **Verify transcript quality** before processing
2. **Update dictionary annually** for latest financial terminology
3. **Calibrate thresholds** based on Scotiabank's industry knowledge
4. **Combine with qualitative analysis** for complete picture
5. **Track changes over time** rather than single-quarter snapshots

## Continuous Improvement

### Quarterly Review Questions

After each quarterly analysis, discuss:

1. Did the rankings align with our market understanding?
2. Were there any surprising results that warrant investigation?
3. Are the example sentences highlighting the right themes?
4. Should we add or remove banks from analysis?
5. How can we better integrate these insights into decision-making?

### Enhancement Opportunities

**Short-term (1-3 months):**
- Automate PDF downloads from known URLs
- Create PowerPoint export functionality
- Build historical trend database

**Medium-term (3-6 months):**
- Develop custom dashboard in Tableau/Power BI
- Integrate with existing competitive intelligence systems
- Add statistical significance testing

**Long-term (6-12 months):**
- Expand to analyze analyst reports and news articles
- Build predictive models using historical sentiment data
- Create real-time monitoring for material events

## Success Metrics

Track these metrics to evaluate tool effectiveness:

**Adoption Metrics:**
- Number of quarterly analyses completed
- Number of stakeholders receiving reports
- Frequency of results cited in strategy documents

**Quality Metrics:**
- Accuracy of rankings vs. market perceptions
- User satisfaction scores from strategy team
- Time saved vs. manual analysis

**Impact Metrics:**
- Strategic decisions informed by analysis
- Early identification of competitor issues
- Improvements in Scotiabank's earnings communications

## Escalation Path

For issues requiring support:

**Technical Issues (installation, errors, bugs):**
- Level 1: Internal IT help desk
- Level 2: Designated technical contact from consulting team
- Level 3: Tool developer (if critical production issue)

**Methodology Questions (interpretation, validation):**
- Level 1: Strategy team lead (trained user)
- Level 2: Consulting team analytics lead
- Level 3: Academic advisors (if research-level question)

**Strategic Questions (application, decisions):**
- Level 1: Corporate strategy leadership
- Level 2: Consulting team project lead
- Level 3: External strategy consultants (if major initiative)

## Contact Information

**Consulting Team Contacts:**
- Project Lead: [Name, Email]
- Technical Lead: [Name, Email]
- Analytics Lead: [Name, Email]

**Scotiabank Sponsors:**
- Strategy Team Sponsor: [Name, Email, Phone]
- IT Support Contact: [Name, Email, Phone]

## Appendix: Quick Reference Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Launch notebook:**
```bash
jupyter notebook multi_bank_sentiment_streamlined.ipynb
```

**Download FinBERT manually (if needed):**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

**Export additional formats:**
```python
# After running analysis
comparison_df.to_excel('results.xlsx', index=False)
comparison_df.to_json('results.json', orient='records')
```

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Next Review:** March 2025 (after Q1 implementation)
