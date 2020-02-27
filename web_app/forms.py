from flask_wtf import FlaskForm
from wtforms import SubmitField,TextAreaField
from wtforms.validators import DataRequired,Length



class PostForm(FlaskForm):
    Question = TextAreaField('Question', validators=[DataRequired()])
    submit = SubmitField('Submit')
