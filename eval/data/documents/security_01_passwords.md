# Password Storage Basics

Passwords should never be stored as plain text. A password database should contain a slow, salted password hash produced by a password-focused algorithm such as Argon2id, scrypt, or bcrypt.

A unique random salt prevents two users with the same password from producing the same stored value. A work factor makes large-scale guessing more expensive. The chosen parameters should be reviewed as hardware becomes faster.

Password hashing is different from encryption. Hashing is intended to be one-way, while encryption is reversible with a key. Password reset tokens should also be random, short-lived, single-use, and stored or compared in a way that limits exposure.

The salt does not need to be secret, but it must be generated independently for every password and stored with the hash. A server should compare a submitted password using the password library's verification function rather than reimplementing the algorithm. Login responses should avoid revealing whether an email address exists, because that information can support account enumeration.

Rate limiting, temporary lockouts, multi-factor authentication, and breached-password checks can reduce the impact of guessing attacks. Passwords should not appear in application logs, analytics events, exception messages, or support tickets. If a password database is suspected to be exposed, rotate application secrets, invalidate sessions, require resets where appropriate, and investigate access logs.