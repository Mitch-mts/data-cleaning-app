"""Cleaning operations, split so each file does one job.

Read ``pipeline.py`` first. It calls the other files in a fixed order.
``services/cleaning_service.py`` re-exports the public functions so the rest
of the app can keep its current imports.
"""
