import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud

# Creating a comprehensive dataset for car rental feedback sentiment analysis
# Columns: review, sentiment_value, category
# sentiment_value: -1 (bad), 0 (neutral), 1 (good)

data = {
    'review': [
        # Positive reviews (sentiment_value = 1)
        'Good performance of car',
        'Excellent customer service and clean vehicle',
        'Very satisfied with the rental experience',
        'Friendly staff and smooth process',
        'Amazing car condition and quick pickup',
        'Outstanding service, will rent again',
        'Perfect vehicle for our trip',
        'Great value for money',
        'Exceptional customer support',
        'Clean car and professional staff',
        'Smooth booking process',
        'Car was in excellent condition',
        'Highly recommend this rental service',
        'Staff was very helpful and courteous',
        'Vehicle exceeded our expectations',
        'Quick and efficient service',
        'Great communication throughout',
        'Car was spotless and well-maintained',
        'Fantastic rental experience overall',
        'Professional and reliable service',

        # Negative reviews (sentiment_value = -1)
        'The car was dirty and smelled bad',
        'Late delivery and poor communication',
        'The vehicle had mechanical issues',
        'Terrible customer service experience',
        'Car broke down during our trip',
        'Hidden fees not mentioned upfront',
        'Staff was rude and unhelpful',
        'Vehicle was not as described',
        'Long wait times at pickup',
        'Car had damage that was not disclosed',
        'Poor fuel efficiency, unexpected costs',
        'Booking system was confusing',
        'Car interior was damaged and dirty',
        'Delayed pickup without notification',
        'Overcharged for minor scratches',
        'Vehicle smelled of cigarettes',
        'Air conditioning was not working',
        'Car had mechanical problems',
        'Unprofessional staff behavior',
        'Worst rental experience ever',

        # Neutral reviews (sentiment_value = 0)
        'Pickup process was okay, nothing special',
        'Car was average, not too bad',
        'Neutral experience, car was fine',
        'Standard service, met basic expectations',
        'Car was decent for the price',
        'Average customer service',
        'Nothing extraordinary, just okay',
        'Car served its purpose',
        'Typical rental experience',
        'Service was adequate',
        'Car was functional but basic',
        'Standard pickup and return process',
        'Average cleanliness and condition',
        'Fair pricing, average service',
        'Car was okay for short trips',
        'Basic vehicle, nothing special',
        'Acceptable service quality',
        'Car met minimum requirements',
        'Standard rental car experience',
        'Neither good nor bad experience'
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
        'Vehicle Quality', 'Customer Service', 'Overall Experience', 'Staff Interaction', 'Vehicle Quality',
        'Customer Service', 'Vehicle Quality', 'Pricing', 'Customer Service', 'Vehicle Quality',
        'Booking Process', 'Vehicle Quality', 'Overall Experience', 'Staff Interaction', 'Vehicle Quality',
        'Process Efficiency', 'Communication', 'Vehicle Quality', 'Overall Experience', 'Customer Service',

        # Categories for negative reviews
        'Vehicle Quality', 'Process Efficiency', 'Vehicle Quality', 'Customer Service', 'Vehicle Quality',
        'Pricing', 'Staff Interaction', 'Vehicle Quality', 'Process Efficiency', 'Vehicle Quality',
        'Vehicle Quality', 'Booking Process', 'Vehicle Quality', 'Process Efficiency', 'Pricing',
        'Vehicle Quality', 'Vehicle Quality', 'Vehicle Quality', 'Staff Interaction', 'Overall Experience',

        # Categories for neutral reviews
        'Process Efficiency', 'Vehicle Quality', 'Overall Experience', 'Customer Service', 'Vehicle Quality',
        'Customer Service', 'Overall Experience', 'Vehicle Quality', 'Overall Experience', 'Customer Service',
        'Vehicle Quality', 'Process Efficiency', 'Vehicle Quality', 'Pricing', 'Vehicle Quality',
        'Vehicle Quality', 'Customer Service', 'Vehicle Quality', 'Overall Experience', 'Overall Experience'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file for WatsonX training
csv_filename = 'car_rental_feedback_sentiment_training.csv'
df.to_csv(csv_filename, index=False)

print(f"Dataset created with {len(df)} reviews")
print(f"Sentiment distribution:")
print(df['sentiment_value'].value_counts().sort_index())
print(f"\nFile saved as: {csv_filename}")

# Display first few rows
print("\nFirst 10 rows of the dataset:")
print(df.head(10))
