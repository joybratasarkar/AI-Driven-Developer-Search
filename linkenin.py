import requests

# Function to get company logo from Clearbit
def get_company_logo(company_name):
    try:
        url = f'https://logo.clearbit.com/{company_name}.com'
        response = requests.get(url)
        
        if response.status_code == 200:
            download_image(response.url, 'company_logo.png')
            print(f"Company logo found: {response.url}")
        else:
            print("Company logo not found.")
    
    except Exception as e:
        print(f"Error occurred while retrieving the company logo: {str(e)}")

# Helper function to download and save images
def download_image(image_url, filename):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename} successfully!")
    else:
        print(f"Failed to download {filename}.")

# Example usage
company_name = input("Enter company name  ").lower()
get_company_logo(company_name)
