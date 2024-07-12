import streamlit as st
import streamlit_authenticator as stauth
import yaml

# Configuration for the authentication
config = {
    'credentials': {
        'usernames': {
            'user1': {
                'name': 'User One',
                'password': 'password123'  # Replace with a hashed password
            },
            'user2': {
                'name': 'User Two',
                'password': 'password456'  # Replace with a hashed password
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'random_key',  # Replace with a secure random key
        'name': 'streamlit-auth'
    },
    'preauthorized': {
        'emails': [
            'user1@example.com',
            'user2@example.com'
        ]
    }
}

# Function to hash the passwords
def hash_password(password):
    return stauth.Hasher([password]).generate()[0]

# Hashing the passwords
if __name__ == "__main__":
    for user, details in config['credentials']['usernames'].items():
        config['credentials']['usernames'][user]['password'] = hash_password(details['password'])

# Create an authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Get the username and authentication status
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    st.write(f'Welcome *{name}*')
    st.title('Streamlit App with Login')
    st.write('This is a basic Streamlit app with login functionality.')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

authenticator.logout('Logout', 'sidebar')
