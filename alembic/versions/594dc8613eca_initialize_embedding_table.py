"""initialize embedding table

Revision ID: 594dc8613eca
Revises: 
Create Date: 2023-12-08 11:58:07.214490

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = '594dc8613eca'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = ('postgres',)
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('embeddings',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('track_id', sa.UUID(), nullable=True),
    sa.Column('user_id', sa.UUID(), nullable=True),
    sa.Column('plugin_name', sa.String(), nullable=True),
    sa.Column('plugin_version', sa.String(), nullable=True),
    sa.Column('text', sa.String(), nullable=True),
    sa.Column('embedding', pgvector.sqlalchemy.Vector(), nullable=True),
    sa.ForeignKeyConstraint(['track_id'], ['tracks.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('embeddings')
    # ### end Alembic commands ###
