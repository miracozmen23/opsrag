"""Retrieval services.

Import concrete services from their owning modules. Keeping this initializer
side-effect free prevents service factories from importing each other while a
package is only partially initialized.
"""
