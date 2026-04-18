"""Migrazione da 6 regimi a 4 quadranti macro.

Rimuove: probability_growth, probability_slowdown, probability_recession, probability_recovery
Aggiunge: probability_reflation, probability_deflation
Mantiene: probability_stagflation, probability_goldilocks

Revision ID: a1b2c3d4e5f6
Revises: 6bc24200b4e6
Create Date: 2026-04-13
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '6bc24200b4e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Aggiungi nuove colonne
    op.add_column('regime_classifications', sa.Column(
        'probability_reflation', sa.Float(), nullable=False, server_default='0.0',
        comment='Probabilita regime reflation',
    ))
    op.add_column('regime_classifications', sa.Column(
        'probability_deflation', sa.Float(), nullable=False, server_default='0.0',
        comment='Probabilita regime deflation',
    ))

    # Migra dati: reflation = growth + recovery, deflation = recession + slowdown
    op.execute("""
        UPDATE regime_classifications SET
            probability_reflation = probability_growth + probability_recovery,
            probability_deflation = probability_recession + probability_slowdown
    """)

    # Aggiorna il campo regime per i record esistenti
    op.execute("UPDATE regime_classifications SET regime = 'reflation' WHERE regime IN ('growth', 'recovery')")
    op.execute("UPDATE regime_classifications SET regime = 'deflation' WHERE regime IN ('recession', 'slowdown')")

    # Rimuovi vecchie colonne
    op.drop_column('regime_classifications', 'probability_growth')
    op.drop_column('regime_classifications', 'probability_slowdown')
    op.drop_column('regime_classifications', 'probability_recession')
    op.drop_column('regime_classifications', 'probability_recovery')

    # Rimuovi server_default
    op.alter_column('regime_classifications', 'probability_reflation', server_default=None)
    op.alter_column('regime_classifications', 'probability_deflation', server_default=None)


def downgrade() -> None:
    # Aggiungi vecchie colonne
    op.add_column('regime_classifications', sa.Column(
        'probability_growth', sa.Float(), nullable=False, server_default='0.0',
    ))
    op.add_column('regime_classifications', sa.Column(
        'probability_slowdown', sa.Float(), nullable=False, server_default='0.0',
    ))
    op.add_column('regime_classifications', sa.Column(
        'probability_recession', sa.Float(), nullable=False, server_default='0.0',
    ))
    op.add_column('regime_classifications', sa.Column(
        'probability_recovery', sa.Float(), nullable=False, server_default='0.0',
    ))

    # Migra dati indietro (split approssimativo)
    op.execute("""
        UPDATE regime_classifications SET
            probability_growth = probability_reflation * 0.6,
            probability_recovery = probability_reflation * 0.4,
            probability_recession = probability_deflation * 0.6,
            probability_slowdown = probability_deflation * 0.4
    """)

    op.execute("UPDATE regime_classifications SET regime = 'growth' WHERE regime = 'reflation'")
    op.execute("UPDATE regime_classifications SET regime = 'recession' WHERE regime = 'deflation'")

    # Rimuovi nuove colonne
    op.drop_column('regime_classifications', 'probability_reflation')
    op.drop_column('regime_classifications', 'probability_deflation')
