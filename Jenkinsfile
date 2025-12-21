pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'python3 --version || true'
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running tests (placeholder)'
            }
        }
    }
}
