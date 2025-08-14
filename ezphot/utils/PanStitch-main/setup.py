from setuptools import setup, find_packages

setup(
    name='PanStitch',  # 패키지 이름
    version='1.0.0',  # 버전 번호
    description='A Python package for downloading and stitching Pan-STARRS images using SWarp.',  # 설명
    author='SilverRon',  # 작성자
    author_email='gregorypaek94@gmail.com',  # 작성자 이메일
    url='https://github.com/SilverRon/PanStitch',  # 프로젝트 URL (GitHub 등)
    packages=find_packages(),  # 패키지 디렉토리를 자동으로 찾아줍니다
    install_requires=[  # 필요한 의존성 패키지 목록
        'numpy',
        'astropy',
        'requests',
        'matplotlib'
    ],
    license='MIT',  # 라이선스 종류
    classifiers=[  # 패키지에 대한 추가적인 메타데이터
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',  # Python 버전 요구 사항
    entry_points={  # 실행 스크립트를 정의합니다 (선택 사항)
        'console_scripts': [
            'panstitch=PanStitch.__main__:main',
        ],
    },
)
