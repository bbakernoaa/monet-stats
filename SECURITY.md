# Security Policy

## Supported Versions

The following versions of monet-stats are currently supported with security updates:

| Version | Supported |
| ------- | --------- |
| 0.x.x   | ✅ |

## Reporting a Vulnerability

If you discover a security vulnerability in monet-stats, please report it responsibly by sending an email to arl.webmaster@noaa.gov. Please do not create a public issue for security vulnerabilities.

When reporting a security vulnerability, please include:

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- Any suggested fixes if known

We will acknowledge your report within 48 hours and will work to address the issue as quickly as possible. Once verified, we will work on a fix and release a security update.

## Security Best Practices

### For Users

- Always install monet-stats from official sources (PyPI or official GitHub repository)
- Keep your dependencies up to date
- Regularly check for security updates
- Use virtual environments to isolate dependencies

### For Contributors

- Follow secure coding practices
- Validate all inputs
- Use parameterized queries where applicable
- Follow the principle of least privilege
- Keep dependencies updated

## Dependency Security

We regularly scan dependencies for known vulnerabilities using tools like `safety` and GitHub's dependency review. All dependencies are checked during our CI/CD pipeline.

## Code Review Process

All code changes undergo a security review process that includes:

- Peer review of all pull requests
- Automated security scanning
- Dependency vulnerability checks
- Static code analysis