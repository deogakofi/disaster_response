import process_data

data = process_data.load_data('disaster_messages.csv', 'disaster_categories.csv')
data_clean = process_data.clean_data(data)
process_data.save_data(data_clean, 'emergency')
