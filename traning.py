import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Creating a comprehensive dataset for car rental feedback sentiment analysis
# Columns: review, sentiment_value, category
# sentiment_value: -1 (bad), 0 (neutral), 1 (good)

data = {
    'review': [
        # Positive reviews (sentiment_value = 1)
        'Good performance of car',
        'Excellent customer service and clean vehicle',
        'Car was in great condition and fuel efficient',
        'Smooth reservation process',
        'Pricing was fair and transparent',
        'Outstanding vehicle quality and professional staff',
        'Quick pickup process and well-maintained car',
        'Friendly staff and competitive rates',
        'Clean interior and exterior, great driving experience',
        'Easy online booking and prompt service',
        'Comfortable ride and excellent fuel economy',
        'Professional customer service team',
        'Vehicle was spotless and ran perfectly',
        'Great value for money and reliable service',
        'Hassle-free rental experience from start to finish',
        'Impressive car condition and courteous staff',
        'Efficient check-in process and quality vehicle',
        'Exceeded expectations with service quality',
        'Perfect car for our vacation needs',
        'Highly recommend this rental service',

        # Negative reviews (sentiment_value = -1)
        'The car was dirty and uncomfortable',
        'Late delivery and poor communication',
        'Staff was rude and unhelpful',
        'Vehicle had mechanical issues during rental',
        'Hidden fees not disclosed upfront',
        'Car smelled bad and had stains on seats',
        'Terrible customer service experience',
        'Long wait times and disorganized staff',
        'Overcharged for damages that were pre-existing',
        'Car broke down and no immediate assistance',
        'Unprofessional behavior from rental agents',
        'Vehicle was not as described in booking',
        'Poor maintenance and safety concerns',
        'Billing errors and difficult refund process',
        'Car was not ready at scheduled pickup time',
        'Unclean vehicle with trash left inside',
        'Staff ignored our complaints and concerns',
        'Expensive rates compared to competitors',
        'Vehicle had empty fuel tank upon pickup',
        'Disappointing service quality overall',

        # Neutral reviews (sentiment_value = 0)
        'Pickup process was okay',
        'Average experience, nothing special',
        'Car was functional but not impressive',
        'Standard rental service, met basic needs',
        'Decent vehicle condition, average staff',
        'Acceptable service for the price paid',
        'Car served its purpose for the trip',
        'Neither good nor bad experience',
        'Basic rental service without any issues',
        'Adequate vehicle for short-term use',
        'Standard pickup and return process',
        'Car was clean but had minor wear',
        'Service was okay, nothing remarkable',
        'Average customer service interaction',
        'Vehicle met minimum expectations',
        'Typical rental experience, no surprises',
        'Reasonable rates and standard service',
        'Car was fine for basic transportation',
        'Normal rental process without problems',
        'Satisfactory but unremarkable service'
    ],
    'sentiment_value': [
        # Positive sentiment values (1)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Negative sentiment values (-1)
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        # Neutral sentiment values (0)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    'category': [
        # Categories for positive reviews
        'Vehicle Performance', 'Customer Service', 'Vehicle Condition', 'Booking Process', 'Pricing',
        'Overall Experience', 'Process Efficiency', 'Customer Service', 'Vehicle Condition', 'Booking Process',
        'Vehicle Performance', 'Customer Service', 'Vehicle Condition', 'Pricing', 'Overall Experience',
        'Overall Experience', 'Process Efficiency', 'Customer Service', 'Vehicle Suitability', 'Overall Experience',

        # Categories for negative reviews
        'Vehicle Condition', 'Process Efficiency', 'Customer Service', 'Vehicle Performance', 'Pricing',
        'Vehicle Condition', 'Customer Service', 'Process Efficiency', 'Pricing', 'Vehicle Performance',
        'Customer Service', 'Vehicle Condition', 'Vehicle Performance', 'Pricing', 'Process Efficiency',
        'Vehicle Condition', 'Customer Service', 'Pricing', 'Vehicle Condition', 'Overall Experience',

        # Categories for neutral reviews
        'Process Efficiency', 'Overall Experience', 'Vehicle Performance', 'Overall Experience', 'Customer Service',
        'Overall Experience', 'Vehicle Suitability', 'Overall Experience', 'Overall Experience', 'Vehicle Suitability',
        'Process Efficiency', 'Vehicle Condition', 'Overall Experience', 'Customer Service', 'Vehicle Performance',
        'Overall Experience', 'Pricing', 'Vehicle Suitability', 'Process Efficiency', 'Overall Experience'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file
csv_filename = 'car_rental_feedback_sentiment.csv'
df.to_csv(csv_filename, index=False)

print(f"Dataset created with {len(df)} reviews")
print(f"CSV file saved as: {csv_filename}")
print("\nDataset preview:")
print(df.head(10))

# Display sentiment distribution
print("\nSentiment Distribution:")
sentiment_counts = df['sentiment_value'].value_counts().sort_index()
print(sentiment_counts)
