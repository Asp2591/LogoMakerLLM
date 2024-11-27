// Select elements
const logoInput = document.getElementById('logo_input');
const logoDisplay = document.getElementById('logo_display');
const createLogoBtn = document.getElementById('create_logo_btn');
const downloadLogo = document.getElementById('download_logo');
const hamburger = document.getElementById('hamburger');
const sidebarMenu = document.querySelector('.sidebar_menu');
const closeMenuBtn = document.getElementById('close_menu_btn');
const signupBtn = document.getElementById('signup_btn');
const loginBtn = document.getElementById('login_btn');
const modal = document.getElementById('form_modal');
const formArea = document.getElementById('form_area');
const closeModalBtn = document.querySelector('.close_modal_btn');

// Event listener for logo creation (POST to /generate)
createLogoBtn.addEventListener('click', function() {
    const logoText = logoInput.value;
    const username = "current_user";  // Replace with dynamic username from session

    if (logoText) {
        const requestBody = {
            username: username,
            prompt: logoText
        };

        fetch('http://127.0.0.1:8000/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        })
        .then(response => response.json())
        .then(data => {
            if (data.image_name) {
                const imageUrl = `http://127.0.0.1:8000/images/${data.image_name}`;
                logoDisplay.textContent = logoText;
                downloadLogo.href = imageUrl;
                downloadLogo.textContent = "Download Logo";
            } else {
                console.error('Logo creation failed:', data);
            }
        })
        .catch(error => console.error('Error:', error));
    }
});

// Event listener to open menu
hamburger.addEventListener('click', function() {
    sidebarMenu.classList.toggle('show');
});

// Event listener to close menu
closeMenuBtn.addEventListener('click', function() {
    sidebarMenu.classList.remove('show');
});

// Modal functionality for signup (POST to /signup)
signupBtn.addEventListener('click', function() {
    formArea.innerHTML = `
        <h2>Sign Up</h2>
        <input type="text" placeholder="Enter your username" id="signupUsername"/>
        <input type="email" placeholder="Enter your email" id="signupEmail"/>
        <input type="password" placeholder="Enter your password" id="signupPassword"/>
        <button id="submitSignup">Submit</button>
    `;
    modal.style.display = 'block';

    document.getElementById('submitSignup').addEventListener('click', function() {
        const username = document.getElementById('signupUsername').value;
        const email = document.getElementById('signupEmail').value;
        const password = document.getElementById('signupPassword').value;

        fetch('http://127.0.0.1:8000/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, password })
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert('Signup successful!');
                modal.style.display = 'none';
            } else {
                console.error('Signup failed:', data);
            }
        })
        .catch(error => console.error('Error:', error));
    });
});

// Modal functionality for login (POST to /login)
loginBtn.addEventListener('click', function() {
    formArea.innerHTML = `
        <h2>Login</h2>
        <input type="text" placeholder="Enter your username or email" id="loginIdentifier"/>
        <input type="password" placeholder="Enter your password" id="loginPassword"/>
        <button id="submitLogin">Login</button>
    `;
    modal.style.display = 'block';

    document.getElementById('submitLogin').addEventListener('click', function() {
        const identifier = document.getElementById('loginIdentifier').value;
        const password = document.getElementById('loginPassword').value;

        fetch('http://127.0.0.1:8000/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ identifier, password })
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert('Login successful!');
                modal.style.display = 'none';
            } else {
                console.error('Login failed:', data);
            }
        })
        .catch(error => console.error('Error:', error));
    });
});

// Close modal
closeModalBtn.addEventListener('click', function() {
    modal.style.display = 'none';
});

// Close modal if clicked outside of modal content
window.addEventListener('click', function(e) {
    if (e.target == modal) {
        modal.style.display = 'none';
    }
});