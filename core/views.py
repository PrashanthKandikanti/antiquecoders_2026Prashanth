import re

from django.shortcuts import redirect, render

from .models import ChatQuery
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.models import User
from django.contrib import messages
from .models import UserProfile
from plant_health.services import diagnose_uploaded_image, format_prediction_for_chat

GREETING_MESSAGES = {
    "hi",
    "hii",
    "hiii",
    "hello",
    "hey",
    "heyy",
    "good morning",
    "good afternoon",
    "good evening",
}


def _normalize_prompt(prompt: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", prompt.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_greeting(prompt: str) -> bool:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return False

    if normalized in GREETING_MESSAGES:
        return True

    return any(normalized.startswith(f"{greeting} ") for greeting in GREETING_MESSAGES)


def _build_response(prompt: str) -> str:
    if _is_greeting(prompt):
        return (
            "Hello! I am here to help with wheat disease detection and crop guidance. "
            "You can upload a plant image or ask a question to get started."
        )

    return f"Simulated AI analysis for: '{prompt}'. (AI integration pending)"

def home(request):
    if not request.session.session_key:
        request.session.create()
    
    session_key = request.session.session_key
    
    if request.method == "POST":
        prompt = request.POST.get('prompt', '').strip()
        uploaded_image = request.FILES.get("image")

        if uploaded_image:
            diagnosis = diagnose_uploaded_image(uploaded_image)
            response = format_prediction_for_chat(diagnosis)
            if not prompt:
                prompt = f"Uploaded image: {uploaded_image.name}"
            ChatQuery.objects.create(session_key=session_key, prompt=prompt, response=response)
        elif prompt:
            response = _build_response(prompt)
            ChatQuery.objects.create(session_key=session_key, prompt=prompt, response=response)
        else:
            return redirect('home')
        return redirect('home')

    history = ChatQuery.objects.filter(session_key=session_key)
    conversation = history.order_by('created_at')
    
    return render(request, 'core/index.html', {'history': history, 'conversation': conversation})


def login_view(request):
    """Custom login view: accepts email or phone as identifier."""
    if request.method == 'POST':
        identifier = request.POST.get('identifier', '').strip()
        password = request.POST.get('password', '')

        user = None
        # try email
        if identifier:
            try:
                user = User.objects.get(email__iexact=identifier)
            except User.DoesNotExist:
                # try phone
                profile = UserProfile.objects.filter(phone_number=identifier).first()
                if profile:
                    user = profile.user

        if user and user.check_password(password):
            auth_login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid credentials. Use registered email or phone and correct password.')

    return render(request, 'registration/login.html')


def signup_view(request):
    """Simple signup: create User + UserProfile with phone number."""
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()
        phone = request.POST.get('phone', '').strip()
        password = request.POST.get('password', '')
        password2 = request.POST.get('password2', '')

        if not email or not password:
            messages.error(request, 'Email and password are required.')
            return render(request, 'registration/signup.html')

        if password != password2:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'registration/signup.html')

        if User.objects.filter(email__iexact=email).exists():
            messages.error(request, 'A user with this email already exists.')
            return render(request, 'registration/signup.html')

        # create username from email local-part (fallback to email)
        username = email.split('@')[0]
        base_username = username
        i = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{i}"
            i += 1

        user = User.objects.create_user(username=username, email=email, password=password, first_name=name)
        profile = UserProfile.objects.create(user=user, phone_number=phone)
        auth_login(request, user)
        return redirect('home')

    return render(request, 'registration/signup.html')


def logout_view(request):
    auth_logout(request)
    return redirect('home')
