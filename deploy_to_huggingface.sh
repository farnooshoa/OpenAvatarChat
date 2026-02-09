#!/bin/bash

# HuggingFace Spaces Deployment Script
# OpenAvatarChat with Speech Recognition Enhancements

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓ ${NC}$1"
}

print_warning() {
    echo -e "${YELLOW}⚠ ${NC}$1"
}

print_error() {
    echo -e "${RED}✗ ${NC}$1"
}

# Header
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   HuggingFace Spaces Deployment Script                       ║"
echo "║   OpenAvatarChat with Speech Recognition Enhancements        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "src/demo.py" ]; then
    print_error "Please run this script from the OpenAvatarChat root directory"
    exit 1
fi

# Get HuggingFace username and space name
print_info "Please enter your HuggingFace username:"
read -p "> " HF_USERNAME

print_info "Please enter your Space name (e.g., openavatarchat-enhanced):"
read -p "> " SPACE_NAME

# Confirm
echo ""
print_warning "You are about to deploy to: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
print_info "Is this correct? (y/n)"
read -p "> " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    print_error "Deployment cancelled"
    exit 0
fi

echo ""
print_info "Starting deployment process..."
echo ""

# Step 1: Check if huggingface-cli is installed
print_info "Step 1/7: Checking HuggingFace CLI..."
if ! command -v huggingface-cli &> /dev/null; then
    print_warning "HuggingFace CLI not found. Installing..."
    pip install huggingface_hub
    print_success "HuggingFace CLI installed"
else
    print_success "HuggingFace CLI found"
fi

# Step 2: Login to HuggingFace
print_info "Step 2/7: Logging in to HuggingFace..."
print_warning "You will be prompted to enter your HuggingFace token"
huggingface-cli login
print_success "Logged in to HuggingFace"

# Step 3: Update submodules
print_info "Step 3/7: Updating git submodules..."
git submodule update --init --recursive --depth 1 || print_warning "Could not update submodules (may already be updated)"
print_success "Submodules updated"

# Step 4: Copy deployment files
print_info "Step 4/7: Copying deployment files..."

# Copy README
if [ -f "README_SPACE.md" ]; then
    cp README_SPACE.md README.md
    print_success "Copied HuggingFace README"
else
    print_warning "README_SPACE.md not found, using existing README.md"
fi

# Copy Dockerfile
if [ -f "Dockerfile.huggingface" ]; then
    cp Dockerfile.huggingface Dockerfile
    print_success "Copied HuggingFace Dockerfile"
else
    print_error "Dockerfile.huggingface not found!"
    print_info "Please create Dockerfile.huggingface first"
    exit 1
fi

# Copy config file
if [ -f "chat_with_groq_enhanced.yaml" ]; then
    cp chat_with_groq_enhanced.yaml config/
    print_success "Copied enhanced configuration"
else
    print_warning "chat_with_groq_enhanced.yaml not found in root, checking config directory..."
    if [ ! -f "config/chat_with_groq_enhanced.yaml" ]; then
        print_error "Enhanced configuration not found!"
        exit 1
    fi
fi

# Step 5: Create .dockerignore if it doesn't exist
print_info "Step 5/7: Checking .dockerignore..."
if [ ! -f ".dockerignore" ]; then
    cat > .dockerignore << 'EOF'
.git
.gitignore
__pycache__/
*.py[cod]
.vscode/
.idea/
.env
.env.local
*.log
logs/
.DS_Store
tmp/
temp/
docs/
*.md
!README.md
EOF
    print_success "Created .dockerignore"
else
    print_success ".dockerignore already exists"
fi

# Step 6: Add remote and commit
print_info "Step 6/7: Setting up git remote..."

# Add HuggingFace remote
if git remote | grep -q "^hf$"; then
    print_warning "Remote 'hf' already exists, updating URL..."
    git remote set-url hf https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME
else
    git remote add hf https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME
    print_success "Added HuggingFace remote"
fi

# Commit changes
print_info "Committing deployment files..."
git add Dockerfile README.md config/ || true
git commit -m "Deploy enhanced OpenAvatarChat to HuggingFace Spaces" || print_warning "Nothing to commit"
print_success "Changes committed"

# Step 7: Push to HuggingFace
print_info "Step 7/7: Pushing to HuggingFace Spaces..."
print_warning "This may take a few minutes..."
echo ""

# Push to main branch (HuggingFace Spaces uses main by default)
git push hf HEAD:main --force

print_success "Deployment complete!"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   Deployment Successful!                                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

print_success "Your Space is being built at:"
echo "  https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo ""

print_info "Next steps:"
echo "  1. Visit your Space URL above"
echo "  2. Go to Settings → Repository secrets"
echo "  3. Add secret:"
echo "     - Name: GROQ_API_KEY"
echo "     - Value: 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"
echo "  4. Wait for build to complete (check Logs tab)"
echo "  5. Test your demo!"
echo ""

print_warning "⚠ Important: Don't forget to add your API key in Space settings!"

echo ""
